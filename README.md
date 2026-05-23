# A Comparison of Multimodal Fusion for Music Emotion Recognition on PMEmo

Compares four fusion strategies (feature-level, model-level, decision-level, cross-modal attention)
plus two unimodal baselines (audio-only, lyrics-only) for continuous valence-arousal prediction
on the PMEmo dataset. Audio encoded with frozen MERT, lyrics with fine-tuned RoBERTa.

## Setup

```bash
git clone https://github.com/irose8000/MER_Multimodal_Fusion.git
cd MER_Multimodal_Fusion
pip install -r requirements.txt
```

## Data

Download PMEmo from https://github.com/HuiZhangDB/PMEmo and place under:

```
data/raw/chorus/        # chorus MP3s
data/raw/lyrics/        # LRC files
data/raw/annotations/   # dynamic_annotations.csv
data/raw/metadata.csv   # contains chorus_start_time
```

## Usage

**1. Preprocess (run once):**

```bash
python scripts/preprocess_data.py
```

**2. Train a condition:**

```bash
python scripts/train.py --config configs/model.yaml          # full training
python scripts/train.py --config configs/model.yaml --debug  # 100-song test
```

Available configs: `audio_only`, `lyrics_only`, `feature`, `model`, `decision`, `cross_modal`

**3. Evaluate:**

```bash
python scripts/evaluate.py --config configs/model.yaml
```

## Results (606-song training)

| Model | CCC Valence | CCC Arousal | PCC Valence | PCC Arousal |
|---|---|---|---|---|
| audio_only | 0.257 | 0.209 | 0.398 | 0.290 |
| lyrics_only | 0.285 | 0.181 | 0.314 | 0.205 |
| feature | 0.266 | 0.180 | 0.312 | 0.212 |
| **model ★** | **0.330** | **0.207** | **0.376** | **0.236** |
| decision | 0.298 | 0.234 | 0.342 | 0.276 |
| cross_modal | 0.147 | 0.086 | 0.410 | 0.272 |
| Chen et al. baseline | 0.150 | 0.120 | 0.230 | 0.170 |

Model-level fusion achieves the highest CCC Valence (0.330), outperforming Chen et al.'s
cross-attention baseline (0.150) by 2.2×. All conditions use identical encoders and training
setup — only the fusion strategy varies.
