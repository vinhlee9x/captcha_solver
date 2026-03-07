# Project Roadmap

## Current State

The core training pipeline is functional end-to-end:
- ✅ Gemini Vision API labeling
- ✅ CRNN model training with CTC loss
- ✅ Real-data-only retraining
- ✅ Confidence-based self-labeling
- ✅ CLI inference (`predict.py`)

## Milestones

### Phase 1 — Foundation (Complete)
- [x] CRNN architecture (`model.py`)
- [x] Training pipeline with mixed datasets (`train.py`)
- [x] Gemini-based labeling (`auto_label.py`)
- [x] Real-data-only retraining (`retrain_real.py`)

### Phase 2 — Data & Accuracy (In Progress)
- [x] Confidence-based self-labeling (`self_label.py`)
- [ ] Achieve >90% character accuracy on live CAPTCHAs
- [ ] Grow labeled dataset to 3,000+ images via iterative self-labeling cycles

### Phase 3 — Hardening
- [ ] Add `pytest` unit tests for `ctc_decode`, `CaptchaDataset`
- [ ] Add model versioning (checkpoint naming with epoch/accuracy)
- [ ] Add character error rate (CER) logging during training
- [ ] Add validation split during training to catch overfitting

### Phase 4 — Serving
- [ ] Re-add FastAPI inference server (`main.py`)
- [ ] Dockerize using a slim PyTorch image
- [ ] Secure the API (API key header or IP allowlist)
- [ ] Add CAPTCHA collection script back for ongoing data gathering

## Known Issues & Technical Debt

| Issue | Priority | Notes |
|---|---|---|
| `python-dotenv` missing from `requirements.txt` | High | `auto_label.py` requires it for `GEMINI_API_KEY` |
| No validation split during training | Medium | Overfitting risk; hold out ~10% of labeled data |
| `predict.py` loads model at module import time | Low | Slows CLI startup; refactor to lazy load if needed |
| `tests/` contains only image assets | Low | No automated test suite exists yet |
