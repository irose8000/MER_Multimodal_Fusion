"""
Implement CCC and PCC as standalone functions operating on 1D tensors
so they can be used consistently across training, validation, and evaluation
"""

import torch

def compute_ccc(y_hat, y):
  mu_yhat = y_hat.mean()
  mu_y    = y.mean()

  sigma_yhat = y_hat.std()
  sigma_y    = y.std()

  covariance = ((y_hat - mu_yhat) * (y - mu_y)).mean()

  ccc = (2 * covariance) / (
      sigma_yhat ** 2 + sigma_y ** 2 + (mu_yhat - mu_y) ** 2 + 1e-8
  )
  return ccc.item()

def compute_pcc(y_hat, y):
  """
  PCC measures linear correlation only, without penalizing mean or variance
  shifts. It is included as a secondary metric for comparison with prior
  PMEmo literature.
  """
  yhat_centered = y_hat - y_hat.mean()
  y_centered    = y - y.mean()

  numerator   = (yhat_centered * y_centered).sum()
  denominator = torch.sqrt(
      (yhat_centered ** 2).sum() * (y_centered ** 2).sum()
  ) + 1e-8

  return (numerator / denominator).item()  
