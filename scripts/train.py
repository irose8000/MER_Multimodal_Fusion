"""
Training entry point for a single MMER fusion condition.
use by calling python scripts/train.py --config configs/audio_only.yaml
(or any other fusion condition besides audio_only)
"""

import os
import sys
import argparse
import torch
import yaml

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloaders
from src.training.trainer import train

def load_config(config_path):
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    #override fusion condition
    with open(config_path, "r") as f:
        override = yaml.safe_load(f)
    
    #make sure learning rates aren't interpretted as strings
    config["training"]["lr"] = float(config["training"]["lr"])
    config["training"]["roberta_lr"] = float(config["training"]["roberta_lr"])

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

  print(f"Training condition: {fusion_type} on {device}")

  #no need to save the test dataloader since this is train.py
  train_loader, val_loader, _ = get_dataloaders(
      processed_dir = config["data"]["processed_dir"],
      batch_size       = config["dataloader"]["batch_size"],
      num_workers      = config["dataloader"]["num_workers"],
      seed             = config["dataloader"]["seed"],
      max_songs        = 100 if args.debug else None
  )

  model = train(
      fusion_type    = fusion_type,
      train_loader   = train_loader,
      val_loader     = val_loader,
      device         = device,
      num_epochs     = config["training"]["num_epochs"],
      lr             = config["training"]["lr"],
      roberta_lr     = config["training"]["roberta_lr"],
      lambda_mse     = config["training"]["lambda_mse"],
      patience       = config["training"]["patience"],
      checkpoint_dir = config["training"]["checkpoint_dir"],
  )

if __name__ == "__main__":
  main()