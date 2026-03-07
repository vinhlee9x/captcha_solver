# CAPTCHA Solver

A CRNN (Convolutional Recurrent Neural Network) model for solving image-based CAPTCHAs, trained with CTC loss. Includes a full pipeline from AI-assisted labeling to self-labeling and CLI inference.

## Features

- **CRNN model**: CNN + Bidirectional LSTM with CTC decoding
- **Multi-source training**: Combines synthetic and real labeled data
- **AI-assisted labeling**: Uses Gemini Vision API to bootstrap ground truth
- **Self-labeling pipeline**: Trained model labels remaining unlabeled images by confidence threshold
- **CLI inference**: Predict a single CAPTCHA image from the command line

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv  # required by auto_label.py, not yet in requirements.txt
```

### 2. Configure environment

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

### 3. Label real CAPTCHAs with Gemini

Place unlabeled images as `unlabeled_<index>.png` in `real_dataset/`, then:

```bash
python auto_label.py
```

### 4. Train the model

```bash
python retrain_real.py
```

### 5. Self-label remaining images

```bash
python self_label.py
```

### 6. Predict a single image

```bash
python predict.py tests/test_real.jpeg
# Output: ab3f9
```

## Dataset Layout

```
real_dataset/
  unlabeled_00001.png    # raw collected CAPTCHAs
  labeled/               # accepted labeled images (<label>_<index>.png)
  rejected/              # low-confidence or invalid predictions
```

## Character Set

Lowercase a–z and digits 0–9 (36 characters + CTC blank token = 37 classes).

## Tech Stack

- Python 3.x, PyTorch, torchvision, Pillow
- Gemini Vision API (`gemini-2.5-flash`)
- `python-dotenv` for secrets management

## Training Workflow

```
auto_label.py  →  seed labeled dataset
retrain_real.py  →  trained model checkpoint
self_label.py  →  more labeled data
retrain_real.py  →  improved model
(repeat)
```
