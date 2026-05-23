"""
Full MMER model combining audio encoder, lyric encoder, fusion module,
and regression head into a single nn.Module.
"""

import torch
import torch.nn as nn

from src.models.encoders import AudioEncoder, LyricEncoder
from src.models.fusion import get_fusion_module, DecisionLevelFusion
from src.models.regression_head import RegressionHead

class MMERModel(nn.Module):
    """
    1. AudioEncoder  (frozen MERT)
    2. LyricEncoder  (fine-tunable RoBERTa)
    3. Fusion module (one of: feature, model, decision, cross_modal)
    4. RegressionHead (shared MLP + Sigmoid)
    """

    def __init__(self, fusion_type, model_dim=768):
      super().__init__()

      #2 baseline experiments with no fusion
      self.is_audio_only = fusion_type == "audio_only"
      self.is_lyrics_only = fusion_type == "lyrics_only"
      if self.is_audio_only:
        self.audio_encoder = AudioEncoder()
        self.lyric_encoder = None
        self.fusion = None
      elif self.is_lyrics_only:
        self.lyric_encoder = LyricEncoder()
        self.audio_encoder = None
        self.fusion = None

      else:        
        self.fusion_type = fusion_type
        self.audio_encoder = AudioEncoder()
        self.lyric_encoder = LyricEncoder()
        self.fusion = get_fusion_module(fusion_type, model_dim=model_dim)

      # Decision-level fusion produces predictions directly, so no regression head needed
      self.is_decision_level = isinstance(self.fusion, DecisionLevelFusion)
      if not self.is_decision_level:
          self.regression_head = RegressionHead(model_dim=model_dim)
      
    def forward(self, audio, lyrics):
      """
      Args:
          audio:  [B, 1, num_samples] mono waveform at 24kHz
          lyrics: list of B lyric strings
      Returns:
          y_hat:  [B, 2] predicted [valence, arousal] in [0, 1]
      """
      #2 baseline experiments with no fusion
      if self.is_audio_only:
        H_A = self.audio_encoder(audio)
        z = H_A.mean(dim=1)  # [B, d]
      elif self.is_lyrics_only:
        H_T = self.lyric_encoder(lyrics)
        z = H_T.mean(dim=1)  # [B, d]

      else:
        # Encode each modality independently
        H_A = self.audio_encoder(audio)   # [B, T_A, model_dim]
        H_T = self.lyric_encoder(lyrics)  # [B, L,   model_dim]

        z = self.fusion(H_A, H_T)

      # Decision-level fusion already outputs [B, 2] predictions
      if self.is_decision_level:
          return z

      # All other fusion types output [B, model_dim] — pass through regression head      
      else:
        return self.regression_head(z)  # [B, 2]        






