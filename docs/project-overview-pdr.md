# Project Overview & PDR — CAPTCHA Solver

## Product Vision

An automated CAPTCHA recognition system targeting a specific 5-character alphanumeric CAPTCHA style. The system is self-improving: it bootstraps ground truth with Gemini Vision API, trains a neural network, then uses that network to label more data autonomously.

## Problem Statement

The target CAPTCHA style is incompatible with off-the-shelf OCR tools (e.g., EasyOCR) due to its specific font, noise, and distortion patterns. A domain-specific neural network trained on real samples is required for reliable recognition.

## Target Users

- Developers integrating automated CAPTCHA solving into a workflow
- Internal automation systems needing to authenticate against a protected endpoint

## System Capabilities

| Capability | Implementation |
|---|---|
| CAPTCHA text recognition | CRNN + CTC decoder |
| Ground truth bootstrapping | Gemini Vision API (`gemini-2.5-flash`) |
| Autonomous dataset expansion | Self-labeling pipeline (confidence-filtered) |
| CLI inference | `predict.py` — single image to stdout |

## Data Requirements

- Minimum viable labeled set: ~500 images (for initial training)
- Self-labeling confidence threshold: 70% minimum per-character probability
- Target dataset size: 3,000+ real labeled CAPTCHAs

## Functional Requirements

1. **AI labeling**: Use Gemini Vision to label in bulk with validity filtering (exact 5 chars, a–z/0–9)
2. **Model training**: CRNN with CTC loss; supports synthetic + real mixed datasets
3. **Self-labeling**: Model predicts unlabeled images; accepted by confidence + length gate
4. **CLI inference**: Accept image path, return predicted text to stdout

## Non-Functional Requirements

- Predictions must run on CPU (no GPU required for inference)
- Self-labeling must be auditable: rejected images are preserved in `real_dataset/rejected/`
- All shared configuration must live in `config.py`

## Out of Scope

- Multi-CAPTCHA-type support
- HTTP inference API
- User-facing UI
