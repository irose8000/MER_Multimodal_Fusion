"""
We split the 606 valid song IDs into train/val/test sets at the song level
(70/15/15) to prevent data leakage across splits. 
We then construct a PMEmoDataset for each split and wrap it in a PyTorch DataLoader.
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset import PMEmoDataset, get_valid_song_ids

def collate(batch):
  """
  Custom collate function to handle variable-length lyric strings.
  PyTorch's default collate cannot stack strings, 
  so we collect them as a list and stack tensors normally.
  """
  return {
    "audio": torch.stack([item["audio"] for item in batch]),
    "lyric": [item["lyric"] for item in batch],  # list of strings for tokenizer
    "valence": torch.stack([item["valence"] for item in batch]),
    "arousal": torch.stack([item["arousal"] for item in batch])
}

def get_dataloaders(
  processed_dir,
  batch_size = 32, #number of 0.5s windows per batch
  num_workers = 4, #safe default for colab (enough to overlap data loading with GPU, but not too much to overwhelm CPU)
  seed = 42,
  max_songs = None
):
  """
  Build and return train, val, and test DataLoaders for PMEmo.

  We split at the song level (not the window level) to ensure that all
  0.5s windows from a given song appear in only one split, preventing
  data leakage between train and test.
  """
  # Get the 606 song IDs that have all three modalities
  all_ids = get_valid_song_ids(processed_dir)

  # Split songs, not windows, so all windows from one song stay together.
  # First split off 70% train, then split the remaining 30% evenly into val/test.
  train_ids, temp_ids = train_test_split(all_ids[:max_songs], test_size=0.30, random_state=seed)
  val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=seed)

  print(f"Song Split \nTrain: {len(train_ids)} \n Val: {len(val_ids)} \n Test: {len(test_ids)}")

  #PMEmoDataset for each split
  shared_kwargs = dict(processed_dir = processed_dir)

  train_dataset = PMEmoDataset(song_ids=train_ids, **shared_kwargs)
  val_dataset   = PMEmoDataset(song_ids=val_ids,   **shared_kwargs)
  test_dataset  = PMEmoDataset(song_ids=test_ids,  **shared_kwargs)

  print(f"0.5s Window Split \nTrain: {len(train_dataset)} \n Val: {len(val_dataset)} \n Test: {len(test_dataset)}")

  # Wrap each dataset in a DataLoader
  # Only shuffle the training set
  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      collate_fn=collate
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      collate_fn=collate
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      collate_fn=collate
  )

  return train_loader, val_loader, test_loader

