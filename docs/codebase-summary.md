# Codebase Summary

## Repository Root (`/`)

| File | LOC | Role |
|---|---|---|
| `config.py` | 13 | Central constants: device, charset, paths |
| `model.py` | 45 | CRNN model definition |
| `train.py` | 134 | Mixed-data training pipeline |
| `retrain_real.py` | 66 | Real-data-only retraining |
| `auto_label.py` | 102 | Gemini Vision API labeling |
| `self_label.py` | 113 | Confidence-based self-labeling |
| `predict.py` | 41 | CLI single-image prediction |
| `requirements.txt` | 7 | Python dependencies |

**Total source LOC:** 521 lines

## Module Dependency Graph

```
config.py
  └── model.py
        └── train.py (CaptchaDataset, captcha_collate_fn)
              └── retrain_real.py (imports dataset utilities)
        └── predict.py
        └── self_label.py

auto_label.py (standalone: config + Gemini API)
```

## Key Data Flows

### Labeling Flow
```
[manual collection] → real_dataset/unlabeled_*.png
auto_label.py       → real_dataset/labeled/<text>_<idx>.png  (Gemini Vision)
self_label.py       → real_dataset/labeled/<text>_<idx>.png  (model confidence)
```

### Training Flow
```
real_dataset/labeled/  →  retrain_real.py  →  captcha_crnn_model.pth
my_dataset/ + real_dataset/labeled/  →  train.py  →  captcha_crnn_model.pth
```

### Inference Flow
```
predict.py  →  CLI  →  preprocess  →  CRNN greedy decode  →  stdout
```

## Test Assets (`tests/`)

Contains sample CAPTCHA images for manual inspection during development:
- `tests/test.jpeg`
- `tests/test_real.jpeg`
- `tests/unlabeled_02995.png`

No automated test suite currently exists.

## External Dependencies

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model, training, transforms |
| `Pillow` | Image preprocessing (median filter, autocontrast) |
| `requests` | Gemini Vision API calls in `auto_label.py` |
| `python-dotenv` | Load `.env` for `GEMINI_API_KEY` — **missing from `requirements.txt`** |
| `captcha` | Synthetic CAPTCHA image generation |
| `numpy<2` | Compatibility constraint |
