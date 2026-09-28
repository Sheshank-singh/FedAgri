import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from training.train_centralized import train_centralized


def main():
    parser = argparse.ArgumentParser(description="Run Centralized Baseline Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mobilenetv3", "resnet50", "efficientnet_b0", "convnext_tiny"],
        help="CNN architecture to train centrally"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--quick", action="store_true", help="Quick mode for verification")
    args = parser.parse_args()

    train_centralized(model_name=args.model, config_path=args.config, quick=args.quick)


if __name__ == "__main__":
    main()
