"""AudioDiTBlock: self-attention + cross-attention + FFN with AdaLN modulation.

Ported from drifting/models/generator.py LightningDiTBlock (lines 230-311),
extended with cross-attention sub-layer for text/speaker conditioning.

Original has 6 modulation parameters (self-attn + FFN), we extend to 9
(+cross-attn shift, scale, gate).
"""

from typing import Optional

import torch
import torch.nn as nn

from drift_tts.models.modules import (
    RMSNorm,
    StandardMLP,
    SwiGLUFFN,
    modulate,
)
from drift_tts.models.attention import CrossAttention, SelfAttention


class AudioDiTBlock(nn.Module):
    """Transformer block: Self-Attn → Cross-Attn → FFN, all with AdaLN.

    Compared to the original LightningDiTBlock:
    - Added cross-attention sub-layer with 3 extra AdaLN params (9 total).
    - 1D temporal sequence instead of 2D spatial patches.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Norms
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm_cross = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)

        # Self-attention
        self.attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
        )

        # Cross-attention (text + speaker conditioning)
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )

        # FFN
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            hid_size = int(2 / 3 * mlp_hidden_dim)
            hid_size = (hid_size + 31) // 32 * 32  # Round up to multiple of 32
            self.mlp = SwiGLUFFN(hidden_size, hid_size)
        else:
            self.mlp = StandardMLP(hidden_size, mlp_hidden_dim)

        # AdaLN modulation: 9 parameters (3 sub-layers × {shift, scale, gate})
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        )
        # Zero-init for stability (matches original)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] latent tokens.
            c: [B, D] conditioning vector (AdaLN).
            context: [B, M, D] text + speaker context for cross-attention.
            context_mask: [B, M] optional attention mask.

        Returns:
            [B, T, D]
        """
        # Compute all 9 modulation params at once
        chunks = self.adaLN_modulation(c)  # [B, 9*D]
        (
            shift_msa, scale_msa, gate_msa,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = chunks.chunk(9, dim=1)

        # Self-attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)

        # Cross-attention
        x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.cross_attn(x_norm, context, context_mask)

        # FFN
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x
