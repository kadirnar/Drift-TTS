"""Speaker encoder for zero-shot voice cloning.

Input: DACVAE latent in (B, T_ref, D) format (channels last, D=128).

Path A — Global (AdaLN):
  Ref latent → Project D→proj_dim → Temporal Avg Pool → MLP → speaker_embedding

Path B — Fine-grained (Cross-Attention):
  Ref latent tokens → projected to hidden_size → cross-attention context
"""

import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """Extract speaker embeddings from reference audio DACVAE latents."""

    def __init__(
        self,
        latent_dim: int = 128,
        proj_dim: int = 256,
        hidden_size: int = 1024,
    ):
        super().__init__()
        # Shared projection from DACVAE latent space
        self.latent_proj = nn.Linear(latent_dim, proj_dim)

        # Path A: Global speaker embedding (for AdaLN)
        self.global_mlp = nn.Sequential(
            nn.Linear(proj_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Path B: Fine-grained tokens (for cross-attention)
        self.token_proj = nn.Linear(proj_dim, hidden_size)

    def forward(self, ref_latent: torch.Tensor) -> tuple:
        """
        Args:
            ref_latent: [B, T_ref, D] DACVAE-encoded reference (channels last).

        Returns:
            speaker_emb: [B, hidden_size] global speaker embedding for AdaLN.
            speaker_tokens: [B, T_ref, hidden_size] fine-grained tokens for cross-attn.
        """
        # Project: [B, T, D] → [B, T, proj_dim]
        ref_proj = self.latent_proj(ref_latent)

        # Path A: Global — temporal average pool → MLP
        speaker_emb = self.global_mlp(ref_proj.mean(dim=1))

        # Path B: Fine-grained — project tokens
        speaker_tokens = self.token_proj(ref_proj)

        return speaker_emb, speaker_tokens
