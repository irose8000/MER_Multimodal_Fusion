"""
Fusion modules for Multimodal Music Emotion Recognition.

We implement four fusion strategies from Liyanarachchi et al. as swappable
nn.Module subclasses. All four accept the same inputs and produce the same
output shape, so we can swap them without changing any other part of the pipeline.
"""

import torch
import torch.nn as nn

class FeatureLevelFusion(nn.Module):
  """
  Early fusion
  average pool each modality to a fixed-size vector,
  concatenate, and pass through a linear projection.
  """
  def __init__(self, model_dim=768):
    super().__init__()
    # Project concatenated [d || d] back down to d for a consistent output size
    self.proj = nn.Linear(model_dim * 2, model_dim)

  def forward(self, H_A, H_T):
    """
    H_A: [B, T_A, d] audio embeddings
    H_T: [B, L, d]   lyric embeddings

    z:   [B, d]      fused representation
    """
    # Average pool across T_A (feature vectors representing a 0.5 window) 
    h_A = H_A.mean(dim=1)  # [B, d]
    #Average pool across L (token embeddings)
    h_T = H_T.mean(dim=1)  # [B, d]
    
    # Concatenate along feature dimension and project back to d
    z = torch.cat([h_A, h_T], dim=-1)  # [B, 2d]
    return self.proj(z)                 # [B, d]

class ModelLevelFusion(nn.Module):
  """
  embedding from each encoder is 
  passed through a separate MLP,
  concatenate, and pass through a linear projection.
  """
  def __init__(self, model_dim=768):
    super().__init__()
    self.mlp_audio = nn.Sequential(
      nn.Linear(model_dim, model_dim),
      nn.ReLU(),
      # Dropout 0.1 reduces overfitting by randomly zeroing 10% of activations during training
      nn.Dropout(0.1)
    )
    self.mlp_lyric = nn.Sequential(
      nn.Linear(model_dim, model_dim),
      nn.ReLU(),
      # Dropout 0.1 reduces overfitting by randomly zeroing 10% of activations during training
      nn.Dropout(0.1)
    )
    # Project concatenated representation back to d
    self.proj = nn.Linear(model_dim * 2, model_dim)
  
  def forward(self, H_A, H_T):
    """
    H_A: [B, T_A, d] audio embeddings
    H_T: [B, L, d]   lyric embeddings

    z:   [B, d]      fused representation
    """
    # Pool each modality to [B, d] before passing through its MLP
    h_A = self.mlp_audio(H_A.mean(dim=1))  # [B, d]
    h_T = self.mlp_lyric(H_T.mean(dim=1))  # [B, d]

    z = torch.cat([h_A, h_T], dim=-1)  # [B, 2d]
    return self.proj(z)                 # [B, d]    
    
class DecisionLevelFusion(nn.Module):
  """
  embedding from each encoder is 
  passed through a separate MLP,
  separate linear projection,
  separate sigmoid,
  combine predictions according to learned weight alpha
  """  
  def __init__(self, model_dim = 768):
    super().__init__()
    # Independent prediction heads for each modality
    self.head_audio = nn.Sequential(
        nn.Linear(model_dim, model_dim // 2), # Halve the hidden dimension before the final prediction layer to compress the representation
        nn.ReLU(),
        nn.Linear(model_dim // 2, 2),  # predict [valence, arousal]
        nn.Sigmoid(),
    )
    self.head_lyric = nn.Sequential(
        nn.Linear(model_dim, model_dim // 2), # Halve the hidden dimension before the final prediction layer to compress the representation
        nn.ReLU(),
        nn.Linear(model_dim // 2, 2),  # predict [valence, arousal]
        nn.Sigmoid(),
    )
    # Learned scalar weight alpha
    self.alpha_logit = nn.Parameter(torch.tensor(0.0))

  def forward(self, H_A, H_T):
    """
    H_A: [B, T_A, d] audio embeddings
    H_T: [B, L, d]   lyric embeddings

    y_hat: [B, 2]    combined valence-arousal prediction (already in [0,1])
    """
    # Average pool across T_A (feature vectors representing a 0.5 window) 
    h_A = H_A.mean(dim=1)  # [B, d]
    #Average pool across L (token embeddings)
    h_T = H_T.mean(dim=1)  # [B, d]

    y_A = self.head_audio(h_A)  # [B, 2]
    y_T = self.head_lyric(h_T)  # [B, 2]

    # Constrain alpha to [0, 1] via Sigmoid so it acts as a valid weight
    alpha = torch.sigmoid(self.alpha_logit)
    return alpha * y_A + (1 - alpha) * y_T  # [B, 2]

class CrossModalFusion(nn.Module):
  """
  embedding from each encoder is
  inputted to cross attention layer where
  Lyric embeddings serve as queries and audio embeddings as keys and values,
  and then passed through a linear projection.
  """  
  def __init__(self, model_dim=768,num_heads = 8):
    super().__init__()
    self.cross_attn = nn.MultiheadAttention(
        embed_dim=model_dim, 
        num_heads=num_heads,
        # Explicitly set key and value dimensions to match audio encoder output size
        kdim=model_dim,
        vdim=model_dim,
        batch_first=True,  # expect [B, seq, d] not [seq, B, d]
    )
    self.proj = nn.Linear(model_dim, model_dim)

  def forward(self, H_A, H_T):
    """
    H_A: [B, T_A, d] audio embeddings
    H_T: [B, L, d]   lyric embeddings

    z:   [B, d]      fused representation
    """
    # No need for pooling until after cross-attention. Cross-attention preserves the full T_A and L sequences so lyrics can selectively attend to relevant audio moments.
    attn_output, _ = self.cross_attn(
        query=H_T,
        key=H_A,
        value=H_A,
    )

    # Average pool the attended sequence to [B, d]
    z = attn_output.mean(dim=1)  # [B, d]
    return self.proj(z)          # [B, d]    



#selecting fusion module by name from config
FUSION_REGISTRY = {
    "feature":    FeatureLevelFusion,
    "model":      ModelLevelFusion,
    "decision":   DecisionLevelFusion,
    "cross_modal": CrossModalFusion,
}
def get_fusion_module(fusion_type, model_dim = 768):
  """
  Instantiate and return a fusion module by name.
  """
  if fusion_type not in FUSION_REGISTRY:
      raise ValueError(f"Unknown fusion type '{fusion_type}'. Choose from {list(FUSION_REGISTRY.keys())}")
  return FUSION_REGISTRY[fusion_type](model_dim=model_dim)
