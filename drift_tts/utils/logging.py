"""Simple logging utilities."""

import logging
import sys
from typing import Dict, Optional

import torch.distributed as dist

logger = logging.getLogger("drift_tts")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def is_rank_zero() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def log_for_0(msg: str, *args, **kwargs) -> None:
    if is_rank_zero():
        logger.info(msg, *args, **kwargs)


class WandbLogger:
    """Lightweight W&B wrapper compatible with the training loop."""

    def __init__(
        self,
        project: str = "drift-tts",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        use_wandb: bool = False,
    ):
        self.use_wandb = use_wandb and is_rank_zero()
        self._step = 0
        if self.use_wandb:
            import wandb

            wandb.init(project=project, entity=entity, name=name)

    def set_step(self, step: int) -> None:
        self._step = step

    def log_dict(self, metrics: Dict, step: Optional[int] = None) -> None:
        s = step if step is not None else self._step
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=s)
        if is_rank_zero():
            summary = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            logger.info("step=%d | %s", s, summary)

    def finish(self) -> None:
        if self.use_wandb:
            import wandb

            wandb.finish()
