"""Learning rate schedule.

Matches the original: warmup → constant (with optional cosine decay).
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 10000,
    total_steps: int = 500000,
    learning_rate: float = 2e-4,
    schedule: str = "const",
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create LR scheduler with warmup.

    Args:
        optimizer: optimizer to schedule.
        warmup_steps: linear warmup steps.
        total_steps: total training steps.
        learning_rate: peak LR (set in optimizer, scheduler returns multiplier).
        schedule: "const" or "cosine".
        min_lr_ratio: minimum LR as fraction of peak (for cosine).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "const":
            return 1.0
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
