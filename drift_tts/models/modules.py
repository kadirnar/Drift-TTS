"""Foundational modules: RMSNorm, SwiGLU FFN, modulate, TimestepEmbedder.

Ported from drifting/models/generator.py (lines 91-148, 455-479).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization (ported from Flax RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            normed = normed * self.weight
        return normed.to(dtype)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: ``x * (1 + scale) + shift``, broadcasting over the token dim."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network (ported from Flax SwiGLUFFN)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class StandardMLP(nn.Module):
    """Standard GELU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="none"))


class TimestepEmbedder(nn.Module):
    """Embed a continuous scalar (e.g. cfg_scale) via frequency + MLP.

    Ported from drifting/models/generator.py lines 455-479.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Match JAX normal init
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] scalar values.

        Returns:
            [B, hidden_size] embeddings.
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)
