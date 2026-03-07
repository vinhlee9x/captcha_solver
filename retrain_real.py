import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CRNN
from config import DEVICE, NUM_CLASSES, MODEL_PATH, REAL_LABELED_DIR
from train import CaptchaDataset, captcha_collate_fn

EPOCHS = 100
LR = 3e-4
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def retrain_real_only():
    dataset = CaptchaDataset(REAL_LABELED_DIR, transform=transform, preprocess=True)
    print(f"Real dataset: {len(dataset)} ảnh")

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=captcha_collate_fn, drop_last=True,
    )

    model = CRNN(img_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Train từ đầu với real data only | LR={LR} | Epochs={EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets, target_lengths in dataloader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            optimizer.zero_grad()
            out = model(images).permute(1, 0, 2)
            input_lengths = torch.full((out.size(1),), out.size(0), dtype=torch.long)
            loss = criterion(out, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        lr_now = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✅ Lưu model tại: {MODEL_PATH}")


if __name__ == "__main__":
    retrain_real_only()
