import os
from dotenv import load_dotenv
load_dotenv()
import re
import time
import shutil
import base64
import requests
from PIL import Image
from config import REAL_DATASET_DIR, CHARS

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

CAPTCHA_LENGTH = 5
LABELED_DIR = os.path.join(REAL_DATASET_DIR, "labeled")
REJECTED_DIR = os.path.join(REAL_DATASET_DIR, "rejected")

PROMPT = (
    f"This is a CAPTCHA image. Read the text exactly as shown. "
    f"The text is always exactly {CAPTCHA_LENGTH} characters long, "
    f"using only lowercase letters (a-z) and digits (0-9). "
    f"Reply with ONLY the {CAPTCHA_LENGTH} characters, nothing else."
)


def encode_image(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_captcha_with_gemini(img_path):
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": encode_image(img_path),
                    }
                },
            ]
        }]
    }
    resp = requests.post(GEMINI_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r'[^a-z0-9]', '', raw.lower().strip())
    return text


def label_images():
    if not GEMINI_API_KEY:
        print("ERROR: Set GEMINI_API_KEY environment variable first.")
        return

    os.makedirs(LABELED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    unlabeled = sorted(
        f for f in os.listdir(REAL_DATASET_DIR)
        if f.startswith("unlabeled_") and f.endswith(".png")
    )
    print(f"Tìm thấy {len(unlabeled)} ảnh cần label...")

    labeled_count = 0
    rejected_count = 0

    for i, fname in enumerate(unlabeled):
        src = os.path.join(REAL_DATASET_DIR, fname)
        try:
            text = read_captcha_with_gemini(src)
            valid = len(text) == CAPTCHA_LENGTH and all(c in CHARS for c in text)

            if valid:
                dest = os.path.join(LABELED_DIR, f"{text}_{labeled_count:05d}.png")
                shutil.copy(src, dest)
                os.remove(src)
                labeled_count += 1
            else:
                print(f"  Reject {fname}: got '{text}'")
                shutil.move(src, os.path.join(REJECTED_DIR, fname))
                rejected_count += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"  Lỗi với {fname}: {e}")
            rejected_count += 1
            time.sleep(1)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(unlabeled)}] OK: {labeled_count} | Loại: {rejected_count}")

    print(f"\nHoàn tất! Labeled: {labeled_count} | Rejected: {rejected_count}")
    print(f"Ảnh đã label tại: {LABELED_DIR}")


if __name__ == "__main__":
    label_images()
