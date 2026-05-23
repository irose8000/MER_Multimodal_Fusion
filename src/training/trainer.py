"""
MMER Training Loop
"""

import torch
import os
import numpy as np

from torch.utils.data import DataLoader
from src.models.model import MMERModel
from src.training.loss import CombinedLoss
from src.utils.metrics import compute_ccc, compute_pcc

def get_optimizer(model, lr= 1e-4, roberta_lr = 1e-5): #Use a lower lr for RoBERTa to prevent large parameter updates from destroying its pretrained representations during fine-tuning.
  roberta_params = list(model.lyric_encoder.roberta.parameters()) if model.lyric_encoder else []
  
  other_params = [
      p for p in model.parameters()
      if not any(p is rp for rp in roberta_params)
      and p.requires_grad
  ]
  
  param_groups = [
    {"params": other_params,  "lr": lr},
    {"params": roberta_params, "lr": roberta_lr},
  ]

  return torch.optim.AdamW(param_groups, weight_decay=1e-2)

def evaluate(model, loader, loss_function, device):
  """
  Run one pass over a DataLoader and return average loss and CCC/PCC metrics.
  Collect all predictions and targets before computing CCC and PCC to
  ensure accurate corpus-level statistics rather than batch-level averages.
  """
  model.eval()
  all_v_hat, all_a_hat = [], []
  all_v,     all_a     = [], []
  total_loss = 0.0

  with torch.no_grad():
    for batch in loader:
      audio   = batch["audio"].to(device)
      lyrics  = batch["lyric"]
      valence = batch["valence"].to(device)
      arousal = batch["arousal"].to(device)

      y_hat = model(audio, lyrics)
      loss, _ = loss_function(y_hat, valence, arousal)
      total_loss += loss.item()

      all_v_hat.append(y_hat[:, 0].cpu())
      all_a_hat.append(y_hat[:, 1].cpu())
      all_v.append(valence.cpu())
      all_a.append(arousal.cpu())

  # Concatenate across all batches before computing metrics
  all_v_hat = torch.cat(all_v_hat)
  all_a_hat = torch.cat(all_a_hat)
  all_v     = torch.cat(all_v)
  all_a     = torch.cat(all_a)

  return {
      "loss":        total_loss / len(loader),
      "ccc_valence": compute_ccc(all_v_hat, all_v),
      "ccc_arousal": compute_ccc(all_a_hat, all_a),
      "pcc_valence": compute_pcc(all_v_hat, all_v),
      "pcc_arousal": compute_pcc(all_a_hat, all_a),
  }

def train(
    fusion_type,
    train_loader,
    val_loader,
    device,
    num_epochs= 50,
    lr = 1e-4,
    roberta_lr = 1e-5,
    lambda_mse = 0.1,
    patience = 5, # Stop training if validation CCC does not improve for 5 consecutive epochs
    checkpoint_dir = "logs/checkpoints",
):
  """
  Train one MMER condition end-to-end with early stopping on validation CCC.
  """
  os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_path = os.path.join(checkpoint_dir, f"{fusion_type}_best.pt")

  model = MMERModel(fusion_type=fusion_type).to(device)
  loss_function = CombinedLoss(lambda_mse=lambda_mse)
  optimizer = get_optimizer(model, lr=lr, roberta_lr=roberta_lr)

  best_val_ccc = -float("inf")
  epochs_without_improvement = 0

  for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0

    for batch in train_loader:
      audio   = batch["audio"].to(device)
      lyrics  = batch["lyric"]
      valence = batch["valence"].to(device)
      arousal = batch["arousal"].to(device)

      optimizer.zero_grad()
      y_hat = model(audio, lyrics)
      loss, _ = loss_function(y_hat, valence, arousal)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation 
    val_metrics = evaluate(model, val_loader, loss_function, device)

    # Monitor average CCC across valence and arousal
    val_ccc = (val_metrics["ccc_valence"] + val_metrics["ccc_arousal"]) / 2

    print(
      f"[{fusion_type}] Epoch {epoch+1:03d} "
      f"Train Loss: {train_loss:.4f} "
      f"Val CCC V: {val_metrics['ccc_valence']:.4f}"
      f"Val CCC A: {val_metrics['ccc_arousal']:.4f}"
      f"Val CCC Avg: {val_ccc:.4f}"
    )

    # Early stopping
    if val_ccc > best_val_ccc:
        best_val_ccc = val_ccc
        epochs_without_improvement = 0
        # Save best model weights
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  -> New best checkpoint saved (val CCC avg: {val_ccc:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement ({epochs_without_improvement}/{patience})")

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break
  
  # Restore best checkpoint before returning
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  print(f"Training complete. Best val CCC avg: {best_val_ccc:.4f}")
  return model

