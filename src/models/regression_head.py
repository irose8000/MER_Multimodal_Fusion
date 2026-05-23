"""
Regression head shared across all fusion conditions (except decision-level,
which produces predictions directly and bypasses this module).
"""

import torch
import torch.nn as nn

class RegressionHead(nn.Module):
  """
  Shared MLP regression head for valence and arousal prediction.
  """

  def __init__(self, model_dim = 768):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(model_dim, model_dim // 2), # Halve the hidden dimension before the final prediction layer to compress the representation
      nn.ReLU(),
      nn.Dropout(0.1), # Dropout 0.1 reduces overfitting by randomly zeroing 10% of activations during training
      nn.Linear(model_dim // 2, 2),
      # Sigmoid constrains predictions to [0, 1] matching PMEmo label range
      nn.Sigmoid(),
    )
  
  def forward(self, z: torch.Tensor) -> torch.Tensor:
    return self.mlp(z)

