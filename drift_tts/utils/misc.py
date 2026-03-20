"""Miscellaneous utilities."""

from pathlib import Path

import yaml


class EasyDict(dict):
    """Dict with attribute access (recursive)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def get(self, key, default=None):
        val = super().get(key, default)
        return val


def _to_easydict(obj):
    if isinstance(obj, dict):
        return EasyDict({k: _to_easydict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_easydict(v) for v in obj]
    return obj


def load_config(path: str) -> EasyDict:
    """Load YAML config and return as nested EasyDict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_easydict(raw)
