"""AudioDiTGen: Main TTS generator model.

Adapted from drifting DitGen + Irodori-TTS architecture insights.

Key design (matching Irodori-TTS DACVAE integration):
- DACVAE latent_dim = 128 (inferred from model)
- Internal format: (B, T, D) — channels last
- Patching: (B, T, 128) → (B, T//patch, 128*patch) — no projection needed
- With patch_size=8: patched_dim = 1024 = hidden_size
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from drift_tts.codec.dacvae import patchify_latent, unpatchify_latent
from drift_tts.models.dit_block import AudioDiTBlock
from drift_tts.models.modules import RMSNorm, TimestepEmbedder, modulate
from drift_tts.models.text_encoder import TextEncoder
from drift_tts.models.speaker_encoder import SpeakerEncoder


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """1D sinusoidal positional encoding → (length, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = np.arange(length, dtype=np.float64)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


class FinalLayer(nn.Module):
    """Output projection with AdaLN."""

    def __init__(self, hidden_size: int, out_dim: int, use_rmsnorm: bool = True):
        super().__init__()
        self.norm = RMSNorm(hidden_size) if use_rmsnorm else nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class AudioDiTGen(nn.Module):
    """One-step TTS generator using DiT architecture with drift training.

    Architecture:
    1. Noise (B, T, latent_dim) → patchify → (B, T//ps, patched_dim) → in_proj → (B, N, hidden)
    2. DiT blocks with self-attn, cross-attn (text+speaker), AdaLN (speaker+cfg)
    3. out_proj → unpatchify → (B, T, latent_dim)
    """

    def __init__(
        self,
        # DACVAE latent specs
        latent_dim: int = 128,
        latent_patch_size: int = 8,
        # DiT specs
        hidden_size: int = 1024,
        depth: int = 22,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        # Text encoder (Echo TTS style: UTF-8 byte transformer)
        vocab_size: int = 259,
        text_hidden_size: int = 512,
        text_num_layers: int = 6,
        text_num_heads: int = 8,
        text_intermediate_size: int = 2048,
        # Speaker encoder
        speaker_proj_dim: int = 256,
        # CFG
        cfg_scale_embed: bool = True,
        # Training
        use_gradient_checkpointing: bool = False,
        # Pos embed
        max_seq_tokens: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_patch_size = latent_patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.use_gradient_checkpointing = use_gradient_checkpointing

        patched_dim = latent_dim * latent_patch_size  # e.g. 128*8 = 1024

        # Input/output projections
        self.in_proj = nn.Linear(patched_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, patched_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Positional embedding
        pos_embed = get_1d_sincos_pos_embed(hidden_size, max_seq_tokens)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).unsqueeze(0), persistent=False)

        # Text encoder (Echo TTS style: UTF-8 byte transformer)
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_size=text_hidden_size,
            output_dim=hidden_size,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            intermediate_size=text_intermediate_size,
        )

        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            latent_dim=latent_dim,
            proj_dim=speaker_proj_dim,
            hidden_size=hidden_size,
        )

        # CFG scale embedder
        self.cfg_embedder = TimestepEmbedder(hidden_size) if cfg_scale_embed else None
        self.cfg_norm = RMSNorm(hidden_size) if cfg_scale_embed else None

        # DiT blocks
        self.blocks = nn.ModuleList([
            AudioDiTBlock(
                hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm, use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm, use_rope=use_rope,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patched_dim, use_rmsnorm)

    def _interpolate_pos_embed(self, num_tokens: int) -> torch.Tensor:
        max_tokens = self.pos_embed.shape[1]
        if num_tokens <= max_tokens:
            return self.pos_embed[:, :num_tokens]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=num_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def build_conditioning(self, speaker_emb: torch.Tensor, cfg_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = speaker_emb
        if self.cfg_embedder is not None and cfg_scale is not None:
            cond = cond + self.cfg_norm(self.cfg_embedder(cfg_scale)) * 0.02
        return cond

    def forward(
        self,
        text_tokens: torch.Tensor,
        ref_latent: torch.Tensor,
        target_frames: int,
        cfg_scale: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate DACVAE latent from text and reference audio.

        Args:
            text_tokens: [B, L] UTF-8 byte token IDs.
            ref_latent: [B, T_ref, D] DACVAE-encoded reference (channels last).
            target_frames: number of output latent frames T.
            cfg_scale: [B] optional per-sample CFG scale.
            text_mask: [B, L] text attention mask.
            temperature: noise temperature.

        Returns:
            [B, T, D] predicted DACVAE latent (channels last).
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        # 1. Generate noise in latent space: (B, T, D)
        ps = self.latent_patch_size
        T_padded = ((target_frames + ps - 1) // ps) * ps
        noise = torch.randn(B, T_padded, self.latent_dim, device=device) * temperature

        # 2. Patchify: (B, T, D) → (B, T//ps, D*ps)
        patched = patchify_latent(noise, ps)
        num_tokens = patched.shape[1]

        # 3. Project + pos embed
        tokens = self.in_proj(patched)
        tokens = tokens + self._interpolate_pos_embed(num_tokens)

        # 4. Text encoding
        text_features = self.text_encoder(text_tokens, mask=text_mask)

        # 5. Speaker encoding
        speaker_emb, speaker_tokens = self.speaker_encoder(ref_latent)

        # 6. Cross-attention context
        context = torch.cat([text_features, speaker_tokens], dim=1)
        if text_mask is not None:
            speaker_mask = torch.ones(B, speaker_tokens.shape[1], device=device, dtype=text_mask.dtype)
            context_mask = torch.cat([text_mask, speaker_mask], dim=1)
        else:
            context_mask = None

        # 7. AdaLN conditioning
        cond = self.build_conditioning(speaker_emb, cfg_scale)

        # 8. DiT blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, cond, context, context_mask, use_reentrant=False)
            else:
                tokens = block(tokens, cond, context, context_mask)

        # 9. Final layer + unpatchify
        tokens = self.final_layer(tokens, cond)
        output = unpatchify_latent(tokens, ps, self.latent_dim)  # (B, T_padded, D)

        return output[:, :target_frames]

    def forward_with_cfg(
        self,
        text_tokens: torch.Tensor,
        ref_latent: torch.Tensor,
        target_frames: int,
        cfg_text: float = 2.0,
        cfg_spk: float = 2.0,
        text_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate with dual classifier-free guidance (text + speaker).

        uncond + cfg_text*(text_cond - uncond) + cfg_spk*(spk_cond - uncond)
        """
        B = text_tokens.shape[0]
        device = text_tokens.device

        ps = self.latent_patch_size
        T_padded = ((target_frames + ps - 1) // ps) * ps
        noise = torch.randn(B, T_padded, self.latent_dim, device=device) * temperature
        patched = patchify_latent(noise, ps)
        num_tokens = patched.shape[1]

        tokens_base = self.in_proj(patched) + self._interpolate_pos_embed(num_tokens)

        text_features = self.text_encoder(text_tokens, mask=text_mask)
        null_text = torch.zeros_like(text_features)

        speaker_emb, speaker_tokens = self.speaker_encoder(ref_latent)
        null_speaker_emb = torch.zeros_like(speaker_emb)
        null_speaker_tokens = torch.zeros_like(speaker_tokens)

        def _run_dit(tokens, cond, context, ctx_mask):
            for block in self.blocks:
                tokens = block(tokens, cond, context, ctx_mask)
            return self.final_layer(tokens, cond)

        # Unconditional
        uncond_ctx = torch.cat([null_text, null_speaker_tokens], dim=1)
        uncond_out = _run_dit(tokens_base.clone(), self.build_conditioning(null_speaker_emb), uncond_ctx, None)

        # Text-conditioned
        text_ctx = torch.cat([text_features, null_speaker_tokens], dim=1)
        text_out = _run_dit(tokens_base.clone(), self.build_conditioning(null_speaker_emb), text_ctx, None)

        # Speaker-conditioned
        spk_ctx = torch.cat([null_text, speaker_tokens], dim=1)
        spk_out = _run_dit(tokens_base.clone(), self.build_conditioning(speaker_emb), spk_ctx, None)

        guided = uncond_out + cfg_text * (text_out - uncond_out) + cfg_spk * (spk_out - uncond_out)

        output = unpatchify_latent(guided, ps, self.latent_dim)
        return output[:, :target_frames]
