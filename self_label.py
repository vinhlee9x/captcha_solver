import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from model import CRNN
from config import DEVICE, INT2CHAR, NUM_CLASSES, REAL_DATASET_DIR, MODEL_PATH

CONFIDENCE_THRESHOLD = 0.70
CAPTCHA_LENGTH = 5

LABELED_DIR = os.path.join(REAL_DATASET_DIR, "labeled")
REJECTED_DIR = os.path.join(REAL_DATASET_DIR, "rejected")

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def preprocess(img_path):
    img = Image.open(img_path).convert('L')
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    return transform(img).unsqueeze(0)


def ctc_decode(log_probs):
    """Greedy CTC decode. Returns (text, min_char_confidence)."""
    probs = log_probs[0].exp()  # (T, C) — remove batch dim first
    best = probs.argmax(dim=1).tolist()  # (T,)

    chars = []
    confidences = []
    prev = 0
    for ti, idx in enumerate(best):
        if idx != 0 and idx != prev:
            chars.append(INT2CHAR.get(idx, ''))
            confidences.append(probs[ti, idx].item())
        prev = idx

    text = ''.join(chars)
    min_conf = min(confidences) if confidences else 0.0
    return text, min_conf


def self_label():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model không tìm thấy tại: {MODEL_PATH}")
        print("   Hãy train model trước bằng: python train.py")
        return

    os.makedirs(LABELED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    model = CRNN(img_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    unlabeled = sorted(
        f for f in os.listdir(REAL_DATASET_DIR)
        if f.startswith("unlabeled_") and f.endswith(".png")
    )
    print(f"Tìm thấy {len(unlabeled)} ảnh cần self-label...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    labeled_count = 0
    rejected_count = 0

    with torch.no_grad():
        for i, fname in enumerate(unlabeled):
            src = os.path.join(REAL_DATASET_DIR, fname)
            try:
                img_tensor = preprocess(src).to(DEVICE)
                log_probs = model(img_tensor)
                text, confidence = ctc_decode(log_probs)

                valid = (
                    len(text) == CAPTCHA_LENGTH
                    and confidence >= CONFIDENCE_THRESHOLD
                )

                if valid:
                    dest = os.path.join(LABELED_DIR, f"{text}_{labeled_count:05d}.png")
                    shutil.copy(src, dest)
                    os.remove(src)
                    labeled_count += 1
                else:
                    shutil.move(src, os.path.join(REJECTED_DIR, fname))
                    rejected_count += 1

            except Exception as e:
                print(f"  Lỗi {fname}: {e}")
                rejected_count += 1

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(unlabeled)}] Labeled: {labeled_count} | Rejected: {rejected_count}")

    print(f"\nHoàn tất! Labeled: {labeled_count} | Rejected: {rejected_count}")
    accept_rate = labeled_count / max(1, labeled_count + rejected_count) * 100
    print(f"Accept rate: {accept_rate:.1f}%")
    print(f"Ảnh đã label tại: {LABELED_DIR}")

    if accept_rate < 30:
        print("\n⚠️  Accept rate thấp. Model cần được train thêm trước khi self-label.")
    elif accept_rate > 80:
        print("\n✅ Accept rate tốt! Có thể retrain với toàn bộ data mới.")


if __name__ == "__main__":
    self_label()
