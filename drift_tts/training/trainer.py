"""Main training loop for Drift-TTS with multi-GPU DDP support.

Ported from drifting/train.py, adapted for PyTorch DDP and audio TTS pipeline.

Launch multi-GPU:
    torchrun --nproc_per_node=N scripts/train.py --config configs/train_500m.yaml
"""

import gc
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from drift_tts.codec.dacvae import DACVAECodec
from drift_tts.data.dataset import CachedLatentDataset, collate_fn, create_dataloader, infinite_sampler
from drift_tts.features.wavlm_extractor import WavLMFeatureExtractor
from drift_tts.loss.drift_loss import drift_loss
from drift_tts.loss.memory_bank import ArrayMemoryBank
from drift_tts.models.generator import AudioDiTGen
from drift_tts.training.scheduler import create_lr_scheduler
from drift_tts.training.train_state import EMAModel
from drift_tts.utils.checkpoint import restore_checkpoint, save_checkpoint
from drift_tts.utils.logging import WandbLogger, log_for_0, setup_logging, is_rank_zero
from drift_tts.utils.misc import EasyDict, load_config


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> Tuple[int, int, torch.device]:
    """Initialize DDP and return (local_rank, world_size, device).

    Works both with torchrun (env vars set) and single-GPU (no env vars).
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return local_rank, world_size, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model from DDP wrapper."""
    return model.module if isinstance(model, DDP) else model


# ---------------------------------------------------------------------------
# CFG sampling
# ---------------------------------------------------------------------------

def sample_cfg_scale(
    batch_size: int,
    cfg_min: float = 1.0,
    cfg_max: float = 3.0,
    neg_cfg_pw: float = 1.0,
    no_cfg_frac: float = 0.3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample CFG scales with power-law distribution."""
    frac = torch.rand(batch_size, device=device)
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = torch.exp(
            torch.log(torch.tensor(cfg_min, device=device))
            + frac * (torch.log(torch.tensor(cfg_max, device=device)) - torch.log(torch.tensor(cfg_min, device=device)))
        )
    else:
        cfg = (cfg_min**pw + frac * (cfg_max**pw - cfg_min**pw)) ** (1 / pw)

    frac2 = torch.rand(batch_size, device=device)
    cfg = torch.where(frac2 < no_cfg_frac, torch.ones_like(cfg), cfg)
    return cfg


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features_for_drift(
    latent: torch.Tensor,
    dacvae: DACVAECodec,
    wavlm: WavLMFeatureExtractor,
) -> Dict[str, torch.Tensor]:
    """Extract WavLM features from DACVAE latent."""
    waveform = dacvae.decode(latent)
    features = wavlm(waveform)
    return features


def compute_drift_training_loss(
    gen_latent: torch.Tensor,
    pos_features: Dict[str, torch.Tensor],
    neg_features: Dict[str, torch.Tensor],
    dacvae: DACVAECodec,
    wavlm: WavLMFeatureExtractor,
    uncond_w: torch.Tensor,
    n_pos: int,
    n_neg: int,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute drift loss in WavLM feature space."""
    gen_features = extract_features_for_drift(gen_latent, dacvae, wavlm)

    total_loss = torch.tensor(0.0, device=gen_latent.device)
    total_info: Dict[str, Any] = {}

    for feat_name in gen_features:
        gen_feat = gen_features[feat_name]
        if feat_name not in pos_features or feat_name not in neg_features:
            continue
        pos_feat = pos_features[feat_name]
        neg_feat = neg_features[feat_name]

        B = pos_feat.shape[0]
        G = gen_feat.shape[0] // B

        gen_r = rearrange(gen_feat, "(b g) t d -> (b t) g d", g=G)
        pos_r = rearrange(pos_feat, "b p t d -> (b t) p d")
        neg_r = rearrange(neg_feat, "b n t d -> (b t) n d")

        BT = gen_r.shape[0]
        weight_neg = repeat(uncond_w, "b -> (b t) k", t=BT // uncond_w.shape[0], k=n_neg)

        loss, info = drift_loss(
            gen=gen_r, fixed_pos=pos_r, fixed_neg=neg_r,
            weight_neg=weight_neg, R_list=R_list,
        )
        total_loss = total_loss + loss.mean()
        for k, v in info.items():
            total_info[f"{k}/{feat_name}"] = v

    return total_loss, total_info


# ---------------------------------------------------------------------------
# Single train step
# ---------------------------------------------------------------------------

def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    pos_features: Dict[str, torch.Tensor],
    neg_features: Dict[str, torch.Tensor],
    dacvae: DACVAECodec,
    wavlm: WavLMFeatureExtractor,
    optimizer: torch.optim.Optimizer,
    ema: EMAModel,
    cfg_min: float = 1.0,
    cfg_max: float = 3.0,
    neg_cfg_pw: float = 1.0,
    no_cfg_frac: float = 0.3,
    gen_per_label: int = 8,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    max_grad_norm: float = 2.0,
) -> Dict[str, float]:
    """Run one generator optimization step."""
    model.train()
    raw_model = unwrap_model(model)
    device = next(raw_model.parameters()).device
    B = batch["token_ids"].shape[0]

    cfg = sample_cfg_scale(B, cfg_min, cfg_max, neg_cfg_pw, no_cfg_frac, device)
    uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, batch.get("n_neg", 16))

    text_tokens = repeat(batch["token_ids"].to(device), "b l -> (b g) l", g=gen_per_label)
    text_mask = repeat(batch["token_mask"].to(device), "b l -> (b g) l", g=gen_per_label)
    ref_latent = repeat(batch["ref_latent"].to(device), "b c t -> (b g) c t", g=gen_per_label)
    target_frames = batch["num_frames"].max().item()
    cfg_expanded = repeat(cfg, "b -> (b g)", g=gen_per_label)

    optimizer.zero_grad()

    gen_latent = model(
        text_tokens=text_tokens,
        ref_latent=ref_latent,
        target_frames=target_frames,
        cfg_scale=cfg_expanded,
        text_mask=text_mask,
    )

    loss, info = compute_drift_training_loss(
        gen_latent=gen_latent,
        pos_features=pos_features,
        neg_features=neg_features,
        dacvae=dacvae,
        wavlm=wavlm,
        uncond_w=uncond_w,
        n_pos=pos_features[list(pos_features.keys())[0]].shape[1],
        n_neg=neg_features[list(neg_features.keys())[0]].shape[1],
        R_list=R_list,
    )

    loss.backward()
    g_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_grad_norm)
    optimizer.step()

    ema.update(raw_model)

    metrics = {
        "loss": loss.item(),
        "g_norm": g_norm.item() if isinstance(g_norm, torch.Tensor) else g_norm,
    }
    for k, v in info.items():
        metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
    return metrics


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def train(config: EasyDict) -> None:
    """Main training loop with DDP multi-GPU support.

    Launch with: torchrun --nproc_per_node=N scripts/train.py --config ...
    """
    setup_logging()
    local_rank, world_size, device = setup_ddp()
    log_for_0("World size: %d, local rank: %d, device: %s", world_size, local_rank, device)

    # Build model
    model_cfg = config.model
    model = AudioDiTGen(
        latent_dim=model_cfg.get("latent_dim", 128),
        latent_patch_size=model_cfg.get("latent_patch_size", 8),
        hidden_size=model_cfg.get("hidden_size", 1024),
        depth=model_cfg.get("depth", 22),
        num_heads=model_cfg.get("num_heads", 16),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        use_qknorm=model_cfg.get("use_qknorm", True),
        use_swiglu=model_cfg.get("use_swiglu", True),
        use_rope=model_cfg.get("use_rope", True),
        use_rmsnorm=model_cfg.get("use_rmsnorm", True),
        vocab_size=model_cfg.get("vocab_size", 259),
        text_hidden_size=model_cfg.get("text_hidden_size", 512),
        text_num_layers=model_cfg.get("text_num_layers", 6),
        text_num_heads=model_cfg.get("text_num_heads", 8),
        text_intermediate_size=model_cfg.get("text_intermediate_size", 2048),
        speaker_proj_dim=model_cfg.get("speaker_proj_dim", 256),
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", True),
    ).to(device)
    log_for_0("Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = unwrap_model(model)

    # Optimizer
    opt_cfg = config.optimizer
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        betas=(opt_cfg.get("adam_b1", 0.9), opt_cfg.get("adam_b2", 0.95)),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    train_cfg = config.train
    scheduler = create_lr_scheduler(
        optimizer,
        warmup_steps=opt_cfg.get("warmup_steps", 10000),
        total_steps=train_cfg.get("total_steps", 500000),
        schedule=opt_cfg.get("lr_schedule", "const"),
    )

    # EMA (on raw model)
    ema = EMAModel(raw_model, decay=train_cfg.get("ema_decay", 0.999))

    # Frozen models
    dacvae = DACVAECodec.load(config.get("dacvae_path", "facebook/dacvae-watermarked"), device=str(device))
    wavlm = WavLMFeatureExtractor(
        model_name=config.get("wavlm_path", "microsoft/wavlm-large"),
    ).to(device)

    # Dataset with DistributedSampler
    dataset = CachedLatentDataset(
        cache_root=config.data.cache_root,
        target_frames=train_cfg.get("chunk_frames", 862),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
    loader = create_dataloader(
        dataset,
        batch_size=train_cfg.get("train_batch_size", 32),
        num_workers=config.data.get("num_workers", 8),
        shuffle=(sampler is None),
    )
    # Override sampler if DDP
    if sampler is not None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_cfg.get("train_batch_size", 32),
            sampler=sampler,
            num_workers=config.data.get("num_workers", 8),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
    train_iter = infinite_sampler(loader)

    # Memory banks (per-process, not shared)
    num_clusters = train_cfg.get("num_clusters", 5000)
    pos_bank = ArrayMemoryBank(num_clusters=num_clusters, max_size=train_cfg.get("positive_bank_size", 32))
    neg_bank = ArrayMemoryBank(num_clusters=1, max_size=train_cfg.get("negative_bank_size", 512))

    # Logger (rank 0 only)
    log_cfg = config.get("logging", {})
    logger = WandbLogger(
        project=log_cfg.get("project", "drift-tts"),
        entity=log_cfg.get("entity", None),
        name=log_cfg.get("name", None),
        use_wandb=log_cfg.get("use_wandb", False),
    )

    # Restore checkpoint
    workdir = config.get("workdir", "runs")
    ckpt = restore_checkpoint(workdir, device="cpu")
    start_step = 0
    if ckpt is not None:
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict(ckpt["ema"])
        start_step = ckpt["step"]
        log_for_0("Resumed from step %d", start_step)

    total_steps = train_cfg.get("total_steps", 500000)
    save_per_step = train_cfg.get("save_per_step", 10000)
    push_per_step = train_cfg.get("push_per_step", 32)
    pos_per_sample = train_cfg.get("pos_per_sample", 32)
    neg_per_sample = train_cfg.get("neg_per_sample", 16)
    R_list = tuple(train_cfg.get("R_list", [0.02, 0.05, 0.2]))
    forward_dict = train_cfg.get("forward_dict", {})

    log_for_0("Starting training from step %d to %d", start_step, total_steps)
    pbar = tqdm(range(start_step, total_steps), initial=start_step, total=total_steps) if is_rank_zero() else range(start_step, total_steps)

    for step in pbar:
        start_time = time.time()
        logger.set_step(step)

        # Set epoch for DistributedSampler
        if sampler is not None:
            sampler.set_epoch(step)

        # Push to memory bank
        n_push = 0
        while n_push < push_per_step:
            batch = next(train_iter)
            latents = batch["latent"].numpy()
            clusters = batch["speaker_cluster"].numpy()
            pos_bank.add(latents, clusters)
            neg_bank.add(latents, clusters * 0)
            n_push += latents.shape[0]

        # Sample from memory banks
        batch = next(train_iter)
        clusters = batch["speaker_cluster"].numpy()
        pos_samples = pos_bank.sample(clusters, pos_per_sample).to(device)
        neg_samples = neg_bank.sample(clusters * 0, neg_per_sample).to(device)

        # Pre-extract features for positive/negative (frozen, no grad)
        with torch.no_grad():
            B, P = pos_samples.shape[:2]
            pos_flat = rearrange(pos_samples, "b p c t -> (b p) c t")
            pos_feats = extract_features_for_drift(pos_flat, dacvae, wavlm)
            pos_feats = {k: rearrange(v, "(b p) t d -> b p t d", p=P) for k, v in pos_feats.items()}

            N = neg_samples.shape[1]
            neg_flat = rearrange(neg_samples, "b n c t -> (b n) c t")
            neg_feats = extract_features_for_drift(neg_flat, dacvae, wavlm)
            neg_feats = {k: rearrange(v, "(b n) t d -> b n t d", n=N) for k, v in neg_feats.items()}

        batch["ref_latent"] = pos_samples[:, 0]
        batch["n_neg"] = neg_per_sample

        metrics = train_step(
            model=model,
            batch=batch,
            pos_features=pos_feats,
            neg_features=neg_feats,
            dacvae=dacvae,
            wavlm=wavlm,
            optimizer=optimizer,
            ema=ema,
            gen_per_label=forward_dict.get("gen_per_label", 8),
            cfg_min=forward_dict.get("cfg_min", 1.0),
            cfg_max=forward_dict.get("cfg_max", 3.0),
            neg_cfg_pw=forward_dict.get("neg_cfg_pw", 1.0),
            no_cfg_frac=forward_dict.get("no_cfg_frac", 0.3),
            R_list=R_list,
            max_grad_norm=train_cfg.get("max_grad_norm", 2.0),
        )

        scheduler.step()
        metrics["lr"] = scheduler.get_last_lr()[0]
        metrics["time"] = time.time() - start_time

        if step % config.get("logging", {}).get("log_every_k", 20) == 0:
            logger.log_dict(metrics)
            if is_rank_zero() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}")

        step_num = step + 1
        if (step_num % save_per_step == 0 or step_num == total_steps) and is_rank_zero():
            save_checkpoint(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict(),
                    "step": step_num,
                    "config": dict(config),
                },
                workdir=workdir,
                step=step_num,
            )

        # Sync all processes at checkpoints
        if world_size > 1 and step_num % save_per_step == 0:
            dist.barrier()

    logger.finish()
    cleanup_ddp()
    log_for_0("Training complete.")
