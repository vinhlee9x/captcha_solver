# System Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────┐
│                   LABELING PIPELINE                  │
│                                                      │
│  [manually collected]                                │
│  real_dataset/unlabeled_*.png                        │
│          │                                           │
│   ┌──────┴──────┐                                    │
│   ▼             ▼                                    │
│ auto_label.py  self_label.py                         │
│ (Gemini API)   (CRNN + confidence ≥70%)              │
│   │             │                                    │
│   └──────┬──────┘                                    │
│          ▼                                           │
│   real_dataset/labeled/                              │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│                  TRAINING PIPELINE                   │
│                                                      │
│  retrain_real.py  →  CRNN  →  captcha_crnn_model.pth │
│  train.py (mixed: synthetic + real)                  │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│                  INFERENCE (CLI)                     │
│                                                      │
│  predict.py  →  preprocess  →  CRNN  →  CTC decode  │
└──────────────────────────────────────────────────────┘
```

## CRNN Model Architecture

```
Input: grayscale image (1, 32, 128)
  │
  ▼
CNN Backbone (4 conv blocks)
  Conv2d(1→64)   + BN + ReLU + MaxPool(2,2)
  Conv2d(64→128) + BN + ReLU + MaxPool(2,2)
  Conv2d(128→256) + BN + ReLU
  Conv2d(256→256) + BN + ReLU + MaxPool(2,1)  ← height-only pool
  Dropout2d(0.25)
  │
  ▼
Feature Map: (B, 256, 4, W')
  Reshape → (B, W', 256×4=1024)
  Linear(1024 → 256)  [map_to_seq]
  Dropout(0.3)
  │
  ▼
Bidirectional LSTM (2 layers, hidden=256)
  Output: (B, W', 512)
  │
  ▼
Linear(512 → NUM_CLASSES=37)
LogSoftmax(dim=2)
  │
  ▼
CTC Loss (training) / Greedy decode (inference)
  Output: 5-character alphanumeric string
```

## CTC Decoding

Greedy decode (used in `predict.py` and `self_label.py`):
1. `argmax` across class dimension at each time step
2. Collapse repeated predictions
3. Remove blank token (index 0)
4. Map remaining indices to characters via `INT2CHAR`

`self_label.py` additionally computes per-character confidence as `min(prob[t, idx])` for accepted characters, applying a 70% threshold gate before moving files to `labeled/`.

## Data Labeling Strategies

| Strategy | Script | Oracle | Throughput |
|---|---|---|---|
| AI labeling | `auto_label.py` | Gemini Vision API | ~2 img/s (rate-limited) |
| Self-labeling | `self_label.py` | Trained CRNN | Batch GPU speed |

## Iterative Training Loop

The project uses a flywheel approach to grow labeled data:

```
1. auto_label.py  →  seed labeled dataset (~500 imgs)
2. retrain_real.py  →  first trained model
3. self_label.py  →  auto-label remaining unlabeled images
4. retrain_real.py or train.py  →  improved model
5. Repeat from step 3 until accept rate degrades
```

## Configuration Reference

| Item | Value |
|---|---|
| Model input size | H=32, W=128 (grayscale) |
| Character set | a–z, 0–9 (36 chars) |
| CTC blank index | 0 |
| CAPTCHA length | 5 characters |
| Default batch size | 64 (`train.py`) / 32 (`retrain_real.py`) |
| Learning rate | 0.001 (`train.py`) / 3e-4 (`retrain_real.py`) |
| LR scheduler | StepLR(step=15, γ=0.5) / CosineAnnealingLR |
| Self-label confidence threshold | 0.70 |
