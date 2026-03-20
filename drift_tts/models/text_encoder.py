"""Text Transformer encoder over UTF-8 bytes.

Inspired by Echo TTS architecture:
- Input: raw UTF-8 bytes (vocab size = 259: 256 bytes + PAD/BOS/EOS)
- Architecture: Transformer encoder with bidirectional attention
- Output: contextual byte representations for cross-attention conditioning

Echo TTS reference specs (2.4B model):
  hidden_size=1280, layers=14, heads=10, intermediate=3328, max_len=768

We scale down proportionally for our ~600M model.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from drift_tts.data.text_processing import VOCAB_SIZE


class TextTransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="none"),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            key_padding_mask: [B, L] True = ignore (PyTorch convention).
        """
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    """Text Transformer encoder over UTF-8 bytes (Echo TTS style).

    Processes raw UTF-8 byte sequences through a transformer encoder
    and projects to cross-attention dimension for the DiT blocks.

    Pipeline:
      UTF-8 bytes → Embedding [B, L, hidden] → Transformer × N → Proj → [B, L, output_dim]
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 512,
        output_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_seq_len: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.layers = nn.ModuleList([
            TextTransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] UTF-8 byte token IDs.
            mask: [B, L] attention mask (1 = valid, 0 = pad).

        Returns:
            [B, L, output_dim] text features for cross-attention.
        """
        B, L = token_ids.shape
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embedding(token_ids) + self.pos_embedding(positions)

        # PyTorch MHA expects key_padding_mask: True = ignore
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # invert: 0=valid→False, 1=pad→True... wait
            # mask: 1=valid, 0=pad → key_padding_mask: True=ignore
            key_padding_mask = ~mask.bool()

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)
        x = self.output_proj(x)
        return x
