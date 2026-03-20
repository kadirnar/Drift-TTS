"""DACVAE encode/decode wrapper.

Follows Irodori-TTS codec.py pattern:
- encode returns (B, T, D) — channels last
- decode expects (B, T, D)
- latent_dim and sample_rate are inferred from the model
"""

from __future__ import annotations

import torch
import torch.nn as nn


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, T, D) → (B, T//patch, D*patch). Tail tokens dropped."""
    if patch_size <= 1:
        return latent
    bsz, seq_len, dim = latent.shape
    usable = (seq_len // patch_size) * patch_size
    latent = latent[:, :usable]
    return latent.reshape(bsz, usable // patch_size, dim * patch_size)


def unpatchify_latent(patched: torch.Tensor, patch_size: int, latent_dim: int) -> torch.Tensor:
    """(B, T_p, D*patch) → (B, T_p*patch, D)."""
    if patch_size <= 1:
        return patched
    return patched.reshape(patched.shape[0], patched.shape[1] * patch_size, latent_dim)


class DACVAECodec:
    """DACVAE codec wrapper (Irodori-TTS style)."""

    def __init__(
        self,
        model: nn.Module,
        sample_rate: int,
        latent_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model = model
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(
        cls,
        repo_id: str = "facebook/dacvae-watermarked",
        device: str = "cuda",
    ) -> DACVAECodec:
        from dacvae import DACVAE

        model = DACVAE.load(repo_id).eval().to(device)
        model_dtype = next(model.parameters()).dtype

        # Infer latent dim
        dummy = torch.zeros(1, 1, 2048, device=device, dtype=model_dtype)
        with torch.inference_mode():
            z = model.encode(dummy)  # (B, D, T)
        latent_dim = int(z.shape[1])

        return cls(
            model=model,
            sample_rate=int(model.sample_rate),
            latent_dim=latent_dim,
            device=torch.device(device),
            dtype=model_dtype,
        )

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        """Encode waveform to latent.

        Args:
            waveform: (B, 1, T) or (1, T) or (B, C, T) audio.
            sample_rate: source sample rate (resamples if needed).

        Returns:
            (B, T_latent, D) latent — channels last.
        """
        import torchaudio

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate is not None and sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        waveform = waveform.to(self.device, dtype=self.dtype)
        z = self.model.encode(waveform)  # (B, D, T)
        return z.transpose(1, 2).contiguous()  # (B, T, D)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to waveform.

        Args:
            latent: (B, T, D) — channels last.

        Returns:
            (B, 1, T_audio) waveform.
        """
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=self.dtype)  # (B, D, T)
        return self.model.decode(z)

    @property
    def hop_length(self) -> int:
        return int(getattr(self.model, "hop_length", 512))

    def frames_for_duration(self, duration_sec: float) -> int:
        return int(duration_sec * self.sample_rate / self.hop_length)
