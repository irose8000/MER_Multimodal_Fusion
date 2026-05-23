"""
Evaluation entry point. Loads a saved checkpoint and reports
CCC and PCC for valence and arousal on the test set.

Usage:
    python scripts/evaluate.py --config configs/audio_only.yaml
"""

import argparse
import yaml
import torch
import os
import sys

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloaders
from src.models.model import MMERModel
from src.training.loss import CombinedLoss
from src.training.trainer import evaluate


def load_config(config_path: str) -> dict:
    """Load and merge condition config on top of base config."""
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    #override fusion condition
    with open(config_path, "r") as f:
        override = yaml.safe_load(f)
   # Merge override into base config
    config.update(override)
    return config


def main():
    # ArgumentParser lets us pass arguments to this script from the command line (e.g. --config configs/audio_only.yaml)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to condition YAML config")
    parser.add_argument("--debug", action="store_true", help="Run on 10 songs only")

    # args.config then holds the string "configs/(fusion type).yaml" for use in load_config().
    args = parser.parse_args()

    config = load_config(args.config)
    fusion_type = config["fusion_type"]

    #using Colab's T4 CUDA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating condition: {fusion_type} on {device}")

    _, _, test_loader = get_dataloaders(
        processed_dir = config["data"]["processed_dir"],
        batch_size       = config["dataloader"]["batch_size"],
        num_workers      = config["dataloader"]["num_workers"],
        seed             = config["dataloader"]["seed"],
        max_songs        = 100 if args.debug else None
    )

    # Load best checkpoint
    checkpoint_path = os.path.join(
        config["training"]["checkpoint_dir"],
        f"{fusion_type}_best.pt"
    )
    model = MMERModel(fusion_type=fusion_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    loss_function = CombinedLoss(lambda_mse=config["training"]["lambda_mse"])
    metrics = evaluate(model, test_loader, loss_function, device)

    print(f"\n--- Test Results: {fusion_type} ---")
    print(f"CCC Valence: {metrics['ccc_valence']:.4f}")
    print(f"CCC Arousal: {metrics['ccc_arousal']:.4f}")
    print(f"PCC Valence: {metrics['pcc_valence']:.4f}")
    print(f"PCC Arousal: {metrics['pcc_arousal']:.4f}")
    print(f"Test Loss:   {metrics['loss']:.4f}")


if __name__ == "__main__":
    main()