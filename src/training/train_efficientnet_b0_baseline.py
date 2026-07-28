import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform

ROOT_DIR = os.getcwd()
os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_csv_path  = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv_path    = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir_path = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

train_dataset = ChestXrayDataset(train_csv_path, images_dir_path, train_transform)
val_dataset   = ChestXrayDataset(val_csv_path,   images_dir_path, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

# ── Baseline: single linear head, no freezing, no staged training ──
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCHS = 15
best_val_acc = 0.0
metrics = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

    # Training
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Val Accuracy: {val_acc:.2f}%")

    metrics.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(ROOT_DIR, "outputs/models/baseline_efficientnet_b0.pth"))
        print("✓ Best model saved.")

pd.DataFrame(metrics).to_csv(
    os.path.join(ROOT_DIR, "outputs/logs/baseline_efficientnet_b0_metrics.csv"), index=False)
print(f"\nBaseline training complete. Best val acc: {best_val_acc:.2f}%")
