"""Training entrypoint for Drift-TTS.

Single GPU:
    uv run python scripts/train.py --config configs/train_500m.yaml

Multi-GPU (DDP):
    uv run torchrun --nproc_per_node=N scripts/train.py --config configs/train_500m.yaml
"""

import argparse

from drift_tts.training.trainer import train
from drift_tts.utils.misc import load_config


def main():
    parser = argparse.ArgumentParser(description="Train Drift-TTS")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_500m.yaml",
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default="runs",
        help="Working directory for checkpoints and logs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["workdir"] = args.workdir
    train(config)


if __name__ == "__main__":
    main()
