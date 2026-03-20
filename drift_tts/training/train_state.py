"""EMA + optimizer state management."""

from typing import Dict

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Ported from the JAX TrainState EMA logic in drifting/train.py.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters: ema = decay * ema + (1 - decay) * param."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Replace model parameters with EMA values, return originals."""
        originals = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                originals[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return originals

    def restore(self, model: nn.Module, originals: Dict[str, torch.Tensor]) -> None:
        """Restore original parameters after EMA application."""
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
