"""
use MERT as a frozen audio encoder and RoBERTa as a fine-tunable lyric encoder.
Freezing MERT keeps its pretrained music representations intact and reduces GPU memory,
while fine-tuning RoBERTa allows it to adapt to emotion-relevant lyric semantics.
"""

import torch
import torch.nn as nn
from transformers import (AutoModel,RobertaModel,RobertaTokenizer)


MERT_MODEL   = "m-a-p/MERT-v1-95M"
ROBERTA_MODEL = "roberta-base"
MAX_LYRIC_LENGTH  = 35  # covers nearly all of PMEmo lyric segments without truncation (see EDA)


class AudioEncoder(nn.Module):
  """
  Frozen MERT encoder for extracting audio representations.

  MERT is pretrained on large-scale music data via self-supervised learning.
  All weights frozen to preserve these representations and reduce variance.
  In our experiment, the encoder is a constant.
  """

  def __init__(self, model_name = MERT_MODEL):
    super().__init__()
    self.mert = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    #Freeze all MERT parameters (no gradient flow)
    for param in self.mert.parameters():
      param.requires_grad = False
  
  def forward(self, waveform):
    """
    input waveform (mono audio at 24kHz)
    returns sequence of audio embeddings

    no tokenization needed, because MERT takes raw waveform samples
    """
    waveform = waveform.squeeze(1)
    outputs = self.mert(waveform)

    # last_hidden_state is [B, T_A, d]:
      #B = batch size, which is set by the dataloaders
      # T_A = number of time steps MERT outputs for a 0.5s window at 24kHz
      # d = hidden dimension, fixed by the version of MERT used
    return outputs.last_hidden_state

class LyricEncoder(nn.Module):
  """
  Fine-tunable RoBERTa encoder for extracting lyric representations.
  """
  def __init__(self, model_name=ROBERTA_MODEL,max_len = MAX_LYRIC_LENGTH):
    super().__init__()
    self.max_length = max_len
    self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
    self.roberta = RobertaModel.from_pretrained(model_name)
    # All RoBERTa parameters are trainable by default.

  def forward(self, lyrics):
    # Tokenize the batch of lyric strings on the same device as the model
    device = next(self.roberta.parameters()).device
    encoded = self.tokenizer(
        lyrics,
        return_tensors="pt", # Return PyTorch tensors rather than lists or NumPy arrays
        padding=True,
        truncation=True,
        max_length=self.max_length,
    ).to(device)

    outputs = self.roberta(**encoded)
    return outputs.last_hidden_state
