"""Drift loss — PyTorch port of drifting/drift_loss.py.

Faithful translation: jax.lax.stop_gradient → .detach(),
jnp → torch, jax.nn.softmax → F.softmax.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise Euclidean distance.

    Args:
        x: [B, N, D]
        y: [B, M, D]

    Returns:
        [B, N, M] distance matrix.
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute drift loss.

    Args:
        gen: [B, C_g, S] generated features.
        fixed_pos: [B, C_p, S] positive (real) features.
        fixed_neg: [B, C_n, S] negative features (optional).
        weight_gen: [B, C_g] per-sample weights (optional).
        weight_pos: [B, C_p] per-sample weights (optional).
        weight_neg: [B, C_n] per-sample weights (optional).
        R_list: temperature radii for kernel function.

    Returns:
        loss: [B] scalar loss per batch element.
        info: dict with scale and per-R loss diagnostics.
    """
    # 1. Defaults & Casting
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = fixed_pos.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = fixed_neg.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)

    # 2. Core Logic (wrapped for stop_gradient)
    def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
        info = {}
        dist = cdist(old_gen_in, targets_in)
        weighted_dist = dist * targets_w_in[:, None, :]  # [B, C_g, C_g+C_n+C_p]
        scale = weighted_dist.mean() / targets_w_in.mean()
        info["scale"] = scale

        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs

        # Normalize distance for kernel
        dist_normed = dist / scale.clamp(min=1e-3)

        # Masking (self-interaction)
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=torch.float32)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))  # [C_g, C_g+C_n+C_p]
        block_mask = block_mask.unsqueeze(0)  # [1, C_g, C_g+C_n+C_p]
        dist_normed = dist_normed + block_mask * mask_val

        # Force Loop
        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R

            affinity = F.softmax(logits, dim=-1)
            aff_transpose = F.softmax(logits, dim=-2)
            affinity = torch.sqrt((affinity * aff_transpose).clamp(min=1e-6))

            affinity = affinity * targets_w_in[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(dim=-1)
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean()

            info[f"loss_{R}"] = f_norm_val

            force_scale = torch.sqrt(f_norm_val.clamp(min=1e-8))
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        return goal_scaled, scale_inputs, info

    # 3. Compute Goal (No Gradients through goal computation)
    with torch.no_grad():
        goal_scaled, scale_inputs, info = calculate_scaled_goal_and_factor(
            old_gen, targets, targets_w
        )

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))

    info = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in info.items()}
    return loss, info
