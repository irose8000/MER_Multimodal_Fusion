"""
Combined CCC + MSE loss for valence-arousal regression.
"""

import torch
import torch.nn as nn

from src.utils.metrics import compute_ccc


class CombinedLoss(nn.Module):
  def __init__(self, lambda_mse=0.1):
      super().__init__()
      self.lambda_mse = lambda_mse
      self.mse = nn.MSELoss()
    
  def forward(self, y_hat, valence_true, arousal_true):
    v_hat = y_hat[:, 0]  # predicted valence
    a_hat = y_hat[:, 1]  # predicted arousal

    # CCC loss per dimension
    ccc_loss_v = 1-compute_ccc(v_hat, valence_true)
    ccc_loss_a = 1-compute_ccc(a_hat, arousal_true)

    # MSE loss per dimension
    mse_v = self.mse(v_hat, valence_true)
    mse_a = self.mse(a_hat, arousal_true)

    # Average across valence and arousal dimensions
    loss_ccc = (ccc_loss_v + ccc_loss_a) / 2
    loss_mse = (mse_v + mse_a) / 2

    # Combined loss: CCC is primary, MSE is auxiliary stabilizer
    loss = loss_ccc + self.lambda_mse * loss_mse

    # Return per-dimension metrics for logging
    metrics = {
        "ccc_valence": 1 - ccc_loss_v, # convert back to CCC from 1-CCC
        "ccc_arousal": 1- ccc_loss_a,
        "mse_valence": mse_v.item(),
        "mse_arousal": mse_a.item(),
    }

    return loss, metrics  




