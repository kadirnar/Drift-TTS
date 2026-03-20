"""Single-step TTS inference (Echo TTS style fixed-length + trim).

The model always generates a fixed number of latent frames (default 862 = 10s).
Trailing silence is automatically detected and trimmed from the output.

Usage:
    from drift_tts.inference import DriftTTSInference

    tts = DriftTTSInference("path/to/checkpoint")
    waveform = tts.generate("Hello world", ref_audio="ref.wav")
"""

from typing import Optional, Union

import torch
import torchaudio

from drift_tts.codec.dacvae import DACVAECodec
from drift_tts.data.audio_processing import load_and_resample
from drift_tts.data.text_processing import UTF8Tokenizer
from drift_tts.models.generator import AudioDiTGen
from drift_tts.training.train_state import EMAModel


# Fixed generation length (10s = 862 DACVAE frames, rounded to patch_size=4)
DEFAULT_TARGET_FRAMES = 860  # 860 = 215 tokens × 4 patch_size


def trim_trailing_silence(
    waveform: torch.Tensor,
    threshold_db: float = -40.0,
    frame_size: int = 2048,
    min_silence_frames: int = 5,
) -> torch.Tensor:
    """Trim trailing silence from waveform.

    Scans from the end and finds where energy consistently exceeds threshold.

    Args:
        waveform: [1, T] or [T] waveform.
        threshold_db: energy threshold in dB below peak.
        frame_size: analysis frame size in samples.
        min_silence_frames: number of consecutive silent frames to trigger trim.

    Returns:
        Trimmed waveform.
    """
    squeeze = waveform.dim() == 1
    if squeeze:
        waveform = waveform.unsqueeze(0)

    audio = waveform[0]
    T = audio.shape[0]
    if T < frame_size:
        return waveform.squeeze(0) if squeeze else waveform

    threshold = 10 ** (threshold_db / 20.0) * audio.abs().max().item()
    if threshold < 1e-8:
        return waveform.squeeze(0) if squeeze else waveform

    # Scan from end
    num_frames = T // frame_size
    last_active = num_frames
    silence_count = 0

    for i in range(num_frames - 1, -1, -1):
        start = i * frame_size
        end = start + frame_size
        frame_energy = audio[start:end].abs().max().item()
        if frame_energy < threshold:
            silence_count += 1
            if silence_count >= min_silence_frames:
                last_active = i + min_silence_frames
        else:
            silence_count = 0
            last_active = i + 1
            break

    end_sample = min(last_active * frame_size + frame_size, T)
    result = waveform[:, :end_sample]
    return result.squeeze(0) if squeeze else result


class DriftTTSInference:
    """One-step TTS inference engine."""

    def __init__(
        self,
        checkpoint_path: str,
        dacvae_path: str = "facebook/dacvae-watermarked",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", {})
        model_cfg = config.get("model", {})

        # Build model
        self.model = AudioDiTGen(
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
        ).to(self.device)

        # Load EMA weights if available
        if "ema" in ckpt:
            ema = EMAModel(self.model)
            ema.load_state_dict(ckpt["ema"])
            ema.apply(self.model)
        else:
            self.model.load_state_dict(ckpt["model"])

        self.model.eval()
        self.dacvae = DACVAECodec.load(dacvae_path, device=str(self.device))
        self.tokenizer = UTF8Tokenizer()

    @torch.no_grad()
    def generate(
        self,
        text: str,
        ref_audio: Union[str, torch.Tensor],
        cfg_text: float = 2.0,
        cfg_spk: float = 2.0,
        temperature: float = 1.0,
        use_dual_cfg: bool = True,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        trim_silence: bool = True,
    ) -> torch.Tensor:
        """Generate speech from text with voice cloning.

        Fixed-length generation (Echo TTS style): always generates
        ``target_frames`` latent frames, then trims trailing silence.

        Args:
            text: input text to synthesize.
            ref_audio: path to reference audio file or [1, T] waveform tensor.
            cfg_text: text classifier-free guidance scale.
            cfg_spk: speaker classifier-free guidance scale.
            temperature: noise temperature.
            use_dual_cfg: use dual CFG (text + speaker) if True.
            target_frames: fixed number of latent frames to generate.
            trim_silence: trim trailing silence from output.

        Returns:
            [1, T_audio] waveform tensor at model sample rate.
        """
        # Encode reference audio
        if isinstance(ref_audio, str):
            ref_waveform = load_and_resample(ref_audio, self.dacvae.sample_rate)
        else:
            ref_waveform = ref_audio
        if ref_waveform.dim() == 1:
            ref_waveform = ref_waveform.unsqueeze(0)
        if ref_waveform.dim() == 2:
            ref_waveform = ref_waveform.unsqueeze(0)  # [1, 1, T]
        ref_waveform = ref_waveform.to(self.device)

        ref_latent = self.dacvae.encode(ref_waveform)  # [1, T_ref, D] channels last

        # Tokenize text
        token_ids, mask = self.tokenizer.batch_encode([text])
        token_ids = token_ids.to(self.device)
        mask = mask.to(self.device)

        # Round target_frames to multiple of temporal_patch_size
        ps = self.model.temporal_patch_size
        target_frames = ((target_frames + ps - 1) // ps) * ps

        # Generate (fixed length)
        if use_dual_cfg:
            latent = self.model.forward_with_cfg(
                text_tokens=token_ids,
                ref_latent=ref_latent,
                target_frames=target_frames,
                cfg_text=cfg_text,
                cfg_spk=cfg_spk,
                text_mask=mask,
                temperature=temperature,
            )
        else:
            latent = self.model(
                text_tokens=token_ids,
                ref_latent=ref_latent,
                target_frames=target_frames,
                text_mask=mask,
                temperature=temperature,
            )

        # Decode to waveform
        waveform = self.dacvae.decode(latent).squeeze(0)  # [1, T_audio]

        # Trim trailing silence
        if trim_silence:
            waveform = trim_trailing_silence(waveform)

        return waveform

    def save_audio(
        self,
        waveform: torch.Tensor,
        path: str,
        sample_rate: int = 44_100,
    ) -> None:
        """Save waveform to file."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(path, waveform.cpu(), sample_rate)
