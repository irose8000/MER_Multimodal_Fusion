"""
Filters to the 606 songs with all three modalities (see EDA),
loads preprocessed 0.5s windows from data/processed/.
"""
import os
import re
import torch
import pandas as pd
import torchaudio
from torch.utils.data import Dataset


def mmss_to_seconds(t):
    """
    Convert mm:ss string to total seconds.
    Used to convert chorus start time.
    """
    minutes, seconds = t.strip().split(":")
    return int(minutes) * 60 + int(seconds)


def parse_lrc(lrc_path):
    """
    Parse an LRC file into a list of (timestamp_seconds, lyric_line) tuples.
    Timestamps are in [mm:ss.xx] format.
    """
    entries = []
    # Use regex pattern to isolate minute, second, and text groups
    pattern = re.compile(r"\[(\d+):(\d+\.\d+)\](.*)")
    with open(lrc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                text = match.group(3).strip()
                timestamp = minutes * 60 + seconds
                entries.append((timestamp, text))
    # Sort lyric entries chronologically by timestamp in case the LRC file is out of order
    return sorted(entries, key=lambda x: x[0])


def get_active_lyric(lrc_entries, query_time):
    """
    Given a list of (timestamp, lyric) pairs and a query time,
    return the lyric line active at that time.
    Returns the last line whose timestamp <= query_time, or empty string if none.
    """
    active = ""
    for ts, text in lrc_entries:
        if ts <= query_time:
            active = text
        else:
            break
    return active


# Inherits from pytorch Dataset class
class PMEmoDataset(Dataset):
    """
    Loads preprocessed 0.5s windows from data/processed/{song_id}.pt.
    Each .pt file contains a list of window dicts with audio, lyric,
    valence, and arousal already preprocessed by preprocess_data.py.

    Args:
        song_ids:      List of song IDs (ints) to include.
        processed_dir: Path to directory containing {song_id}.pt files.
    """

    def __init__(self, song_ids, processed_dir):
        self.windows = []
        for sid in song_ids:
            pt_path = os.path.join(processed_dir, f"{sid}.pt")
            self.windows.extend(torch.load(pt_path))

    def __len__(self):
        return len(self.windows)

    # Loads and returns the preprocessed audio window, active lyric, and valence/arousal labels for a single 0.5s frame
    def __getitem__(self, idx):
        return self.windows[idx]

def get_valid_song_ids(processed_dir):
    """
    Return sorted list of song IDs that have a preprocessed .pt file
    in data/processed/.
    """
    return sorted([
        int(f.replace(".pt", ""))
        for f in os.listdir(processed_dir)
        if f.endswith(".pt")
    ])
