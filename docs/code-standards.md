# Code Standards

## Language & Runtime

- **Python 3.x** (no version pin; recommend ≥ 3.10)
- One virtual environment per project (`venv/`)

## Project Structure

```
captcha_solver/
├── config.py          # All global constants
├── model.py           # CRNN architecture
├── train.py           # Mixed-data training + dataset class
├── retrain_real.py    # Real-data-only training entry point
├── auto_label.py      # Gemini-based AI labeling
├── self_label.py      # Confidence-based self-labeling
├── predict.py         # CLI inference
├── requirements.txt
├── .env               # Secrets (gitignored)
├── real_dataset/
│   ├── unlabeled_*.png
│   ├── labeled/
│   └── rejected/
├── tests/             # Sample CAPTCHA images (not automated tests)
└── docs/
```

## Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `CRNN`, `CaptchaDataset`)
- **Functions**: `snake_case` (e.g., `ctc_decode`, `build_transform`)
- **Constants**: `UPPER_SNAKE_CASE` defined in `config.py`
- **Labeled image filename format**: `<label>_<index:05d>.png` (e.g., `ab3f9_00042.png`)

## Configuration

All shared constants live in `config.py`:
- `DEVICE` — auto-detect CUDA/CPU
- `CHARS` / `CHAR2INT` / `INT2CHAR` — character vocabulary
- `NUM_CLASSES` — CHARS + CTC blank (37 total)
- `MODEL_PATH` — default model checkpoint path
- `DATASET_DIR` / `REAL_DATASET_DIR` / `REAL_LABELED_DIR` — data directories

New constants must be added to `config.py`; avoid hardcoding strings across modules.

## Image Preprocessing

Standard preprocessing pipeline (applied consistently across `predict.py`, `self_label.py`, and `train.py`):
1. Convert to grayscale (`L` mode)
2. Median filter (size=3) — noise reduction
3. Auto-contrast — enhance character visibility
4. Resize to `(32, 128)` — H×W
5. `ToTensor()` + Normalize `mean=0.5, std=0.5`

## Dataset Conventions

- Unlabeled images: `unlabeled_<index:05d>.png` in `real_dataset/`
- Labeled images: `<label>_<index:05d>.png` in `real_dataset/labeled/`
- Rejected images go to `real_dataset/rejected/` (never deleted)
- Label is always extracted from the filename prefix before the first `_`

## Model Checkpointing

- Model saved via `torch.save(model.state_dict(), MODEL_PATH)`
- Loaded with `map_location=DEVICE` to support CPU-only inference

## Error Handling

- Labeling scripts catch exceptions per item and continue, printing the error
- Scripts print progress every N items to avoid silent failures

## Comments Policy

Do not add comments unless the logic is complex, non-obvious, or explains a critical design decision (e.g., the CTC blank token convention, greedy decode deduplication). Prefer expressive naming.
