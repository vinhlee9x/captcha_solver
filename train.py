import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
from model import CRNN
from config import DEVICE, CHAR2INT, NUM_CLASSES, DATASET_DIR, MODEL_PATH, REAL_LABELED_DIR


SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, transform=None, preprocess=False):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')

        if self.preprocess:
            image = image.filter(ImageFilter.MedianFilter(size=3))
            image = ImageOps.autocontrast(image)

        if self.transform:
            image = self.transform(image)

        label_str = img_name.split('_')[0]
        target = [CHAR2INT[c] for c in label_str if c in CHAR2INT]
        target = torch.tensor(target, dtype=torch.long)
        target_length = torch.tensor([len(target)], dtype=torch.long)
        return image, target, target_length


def captcha_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def build_transform(augment=False):
    ops = []
    if augment:
        ops += [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        ]
    ops += [
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    return transforms.Compose(ops)


def train_crnn():
    synthetic_transform = build_transform(augment=False)
    real_transform = build_transform(augment=True)

    datasets = []

    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        synthetic_dataset = CaptchaDataset(img_dir=DATASET_DIR, transform=synthetic_transform)
        print(f"Synthetic data: {len(synthetic_dataset)} ảnh")
        datasets.append(synthetic_dataset)
    else:
        print("Không tìm thấy my_dataset, bỏ qua.")

    if os.path.exists(REAL_LABELED_DIR) and len(os.listdir(REAL_LABELED_DIR)) > 0:
        real_dataset = CaptchaDataset(img_dir=REAL_LABELED_DIR, transform=real_transform, preprocess=True)
        print(f"Real data: {len(real_dataset)} ảnh")
        datasets.append(real_dataset)
    else:
        print("Chưa có real data, bỏ qua.")

    if not datasets:
        print("❌ Không có dataset nào! Hãy tạo dữ liệu trước.")
        return


    combined = ConcatDataset(datasets)
    dataloader = DataLoader(
        combined, batch_size=64, shuffle=True,
        num_workers=4, collate_fn=captcha_collate_fn
    )

    model = CRNN(img_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    model.train()
    num_epochs = 50

    print(f"Bắt đầu huấn luyện trên {len(combined)} ảnh, {num_epochs} epochs...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets, target_lengths in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).permute(1, 0, 2)
            input_lengths = torch.full(
                size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long
            )
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:02d}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {lr:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Hoàn tất! Đã lưu mô hình tại {MODEL_PATH}")


if __name__ == '__main__':
    train_crnn()