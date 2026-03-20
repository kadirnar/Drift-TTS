"""1D Self-Attention + Cross-Attention + RoPE.

Ported from drifting/models/generator.py Attention class (lines 153-228),
adapted from 2D spatial to 1D temporal for audio sequences.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from drift_tts.models.modules import RMSNorm


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D Rotary Positional Embedding.

    Args:
        q, k: [B, N, H, D]

    Returns:
        Rotated q, k with same shape.
    """
    B, N, H, D = q.shape
    half_dim = D // 2
    freqs = (1.0 / (10000 ** (torch.arange(0, half_dim, device=q.device, dtype=dtype) / half_dim)))
    t = torch.arange(N, device=q.device, dtype=dtype)
    freqs = torch.outer(t, freqs)  # [N, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [N, D]

    cos = emb.cos()[None, :, None, :]  # [1, N, 1, D]
    sin = emb.sin()[None, :, None, :]

    def rotate_half(x):
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional qk-norm and RoPE.

    Ported from drifting/models/generator.py Attention (lines 153-228).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        use_rope: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = None
        self.k_norm = None
        if qk_norm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]

        Returns:
            [B, N, C]
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, N, H, D]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            q, k = apply_rope_1d(q, k)

        # Scale and transpose: [B, H, N, D]
        q = q.transpose(1, 2) * (self.head_dim ** -0.5)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention (query attends to key/value from conditioning)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

        self.q_norm = None
        self.k_norm = None
        if qk_norm:
            if use_rmsnorm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
                self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] queries.
            context: [B, M, C] keys/values from text encoder (+ optional speaker tokens).
            context_mask: [B, M] bool mask (True = attend, False = ignore).

        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)  # [B, N, H, D]
        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, M, H, D]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2) * (self.head_dim ** -0.5)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, M, D]
        v = v.transpose(1, 2)  # [B, H, M, D]

        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, M]

        if context_mask is not None:
            # context_mask: [B, M] → [B, 1, 1, M]
            mask = context_mask[:, None, None, :].bool()
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)  # [B, H, N, D]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x
