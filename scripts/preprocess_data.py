"""
Runs once offline to preprocess all 606 PMEmo songs into 0.5s windows.
Saves one .pt file per song containing all windows, lyrics, and labels
so that dataset.py can load preprocessed tensors directly without
touching raw MP3s or LRC files during training.
"""

import os
import sys
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import (
    parse_lrc,
    get_active_lyric,
    mmss_to_seconds,
)

# Paths
ANNOTATIONS_PATH = "data/raw/annotations/dynamic_annotations.csv"
AUDIO_DIR        = "data/raw/chorus"
LYRICS_DIR       = "data/raw/lyrics"
METADATA_PATH    = "data/raw/metadata.csv"
OUTPUT_DIR       = "data/processed"

# Audio settings
TARGET_SR      = 24000
WINDOW_SAMPLES = int(TARGET_SR * 0.5)  # 12000 samples per 0.5s window

def preprocess_song(
    song_id,
    audio_dir,
    lyrics_dir,
    song_anns,
    chorus_start_seconds,
):
    """
    For each annotation frame we:
      1. Extract and resample the corresponding audio window
      2. Convert stereo to mono
      3. Apply dynamic normalization (zero mean, unit variance)
      4. Look up the active lyric using chorus-relative time + chorus start offset

    Returns a list of dicts, one per window, ready to be stacked and saved.
    """
    # Load and preprocess audio once per song
    audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
    waveform, sr = torchaudio.load(audio_path)
    # Stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample to 24kHz if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Parse LRC once per song
    lrc_path = os.path.join(lyrics_dir, f"{song_id}.lrc")
    lrc_entries = parse_lrc(lrc_path)

    windows = []
    
    for frame_idx, row in song_anns.reset_index(drop=True).iterrows(): # Reset index so frame_idx (0, 1, 2...) maps correctly to row after filtering to one song
        valence     = float(row["Valence(mean)"])
        arousal     = float(row["Arousal(mean)"])
        chorus_time = float(row["frameTime"])
        
        # Audio and annotations are both relative to the chorus, so index directly
        #frame_idx is inputted as part of idx as an input to __getitem__
        #self.window_samples will be 12000 samples if sample rate is 24000 Hz (24k samples in 1sec => 12k samples in 0.5sec)
        start = frame_idx * WINDOW_SAMPLES
        end   = start + WINDOW_SAMPLES
        total_samples = waveform.shape[1]
        if end <= total_samples:
            window = waveform[:, start:end]
        else:
            # Pad last window if shorter than expected
            window = waveform[:, start:]
            pad_len = WINDOW_SAMPLES - window.shape[1]
            window = torch.nn.functional.pad(window, (0, pad_len))

        # Dynamic standardization: zero mean, unit variance per window
        # Each 0.5s window is standardized, so a quiet window and a loud window will have the same mean and std dev
        # This prevents the model from learning shortcut loud window = high arousal/valence and vice-versa
        mean = window.mean()
        std  = window.std()
        window = (window - mean) / std if std > 1e-6 else window - mean

        # Lyric alignment
        # Convert chorus-relative time to song-relative time for LRC lookup
        song_time = chorus_start_seconds + chorus_time
        lyric = get_active_lyric(lrc_entries, song_time)

        windows.append({
            "audio":   window,                                        # [1, 12000]
            "lyric":   lyric,                                         # str
            "valence": torch.tensor(valence, dtype=torch.float32),
            "arousal": torch.tensor(arousal, dtype=torch.float32),
        })

    return windows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load metadata
    annotations_df   = pd.read_csv(ANNOTATIONS_PATH)
    chorus_start_df  = pd.read_csv(METADATA_PATH)
    chorus_start_map = {
        row["musicId"]: mmss_to_seconds(row["chorus_start_time"])
        for _, row in chorus_start_df.iterrows()
    }

    # Get valid song IDs
    df_dynamic = pd.read_csv(ANNOTATIONS_PATH)
    annotated_ids = set(df_dynamic["musicId"].unique())
    lyric_ids = {int(f.replace(".lrc", "")) for f in os.listdir(LYRICS_DIR) if f.endswith(".lrc")}
    audio_ids = {int(f.replace(".mp3", "")) for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")}
    valid_ids = sorted(annotated_ids & lyric_ids & audio_ids)
    print(f"Preprocessing {len(valid_ids)} songs...")

    skipped = 0
    for song_id in tqdm(valid_ids):
        output_path = os.path.join(OUTPUT_DIR, f"{song_id}.pt")

        # Skip songs already preprocessed so we can resume if interrupted
        if os.path.exists(output_path):
            continue

        try:
            song_anns = annotations_df[
                annotations_df["musicId"] == song_id
            ]
            # Default to 0.0 if chorus start is missing, treating the song as starting at the beginning
            chorus_start_seconds = float(chorus_start_map.get(song_id, 0.0))

            windows = preprocess_song(
                song_id=song_id,
                audio_dir=AUDIO_DIR,
                lyrics_dir=LYRICS_DIR,
                song_anns=song_anns,
                chorus_start_seconds=chorus_start_seconds,
            )

            # Save all windows for this song as a single .pt file
            torch.save(windows, output_path)

        except Exception as e:
            print(f"Skipping song {song_id}: {e}")
            skipped += 1

    print(f"Done. Skipped {skipped} songs. Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()