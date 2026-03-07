import sys
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from model import CRNN
from config import DEVICE, INT2CHAR, NUM_CLASSES, MODEL_PATH

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

model = CRNN(img_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict(img_path):
    img = Image.open(img_path).convert('L')
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(t).permute(1, 0, 2)

    _, idx = out.max(2)
    chars, prev = [], -1
    for i in idx.view(-1).tolist():
        if i != 0 and i != prev:
            chars.append(INT2CHAR.get(i, ''))
        prev = i
    return ''.join(chars)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)
    print(predict(sys.argv[1]))
