"""Checkpoint save/restore utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from drift_tts.utils.logging import log_for_0


def save_checkpoint(
    state: Dict[str, Any],
    workdir: str,
    step: int,
    keep_last: int = 2,
    keep_every: int = 500000,
) -> str:
    """Save training state and rotate old checkpoints."""
    ckpt_dir = Path(workdir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(state, path)
    log_for_0("Saved checkpoint: %s", path)

    # Rotate: keep last N + every K
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for ckpt in ckpts[:-keep_last]:
        step_num = int(ckpt.stem.split("_")[1])
        if step_num % keep_every != 0:
            ckpt.unlink()
    return str(path)


def restore_checkpoint(
    workdir: str,
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Restore most recent checkpoint if available."""
    ckpt_dir = Path(workdir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return None
    path = ckpts[-1]
    log_for_0("Restoring checkpoint: %s", path)
    return torch.load(path, map_location=device, weights_only=False)
