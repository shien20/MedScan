import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform

ROOT_DIR = os.getcwd()
os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/confusion_matrix"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================
# DATASETS
# =====================================

train_csv_path  = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv_path    = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir_path = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

train_dataset = ChestXrayDataset(train_csv_path, images_dir_path, train_transform)
val_dataset   = ChestXrayDataset(val_csv_path,   images_dir_path, val_transform)

# =====================================
# DATALOADERS (Stage 1: natural distribution, no weighting)
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True  # Stage 1: use natural class distribution
)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =====================================
# MODEL
# =====================================

model = models.efficientnet_b0(weights="IMAGENET1K_V1")

num_features = model.classifier[1].in_features  # 1280

# Stronger classifier head for medical imaging (more capacity)
model.classifier = nn.Sequential(
    nn.Linear(num_features, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 4)
)

model = model.to(device)

# =====================================
# LOSS FUNCTION (standard, no class weighting for Stage 1)
# =====================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# =====================================
# TRAINING SETTINGS (Architecture-specific for lightweight EfficientNet-B0)
# =====================================

STAGE_1_EPOCHS = 8  # Longer Stage 1 to stabilize stronger head (was 5)
STAGE_2_EPOCHS = 10  # Progressive fine-tuning
best_val_acc = 0.0
best_epoch = 0
best_stage = ""
metrics = []
global_epoch = 0

# =====================================
# STAGE 1: Freeze backbone, train head
# =====================================

print("\n" + "="*60)
print("STAGE 1: Training Classification Head (Backbone Frozen)")
print("="*60)

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Fix 3: Lower LR for Stage 1
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=STAGE_1_EPOCHS)

for epoch in range(STAGE_1_EPOCHS):
    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE_1_EPOCHS}] - STAGE 1")

    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc   = 100 * train_correct / train_total
    train_loss /= len(train_loader)
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

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
        'epoch': global_epoch, 'stage': 'Stage 1',
        'train_loss': train_loss, 'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = global_epoch
        best_stage   = "Stage 1"
        torch.save(model.state_dict(),
                   os.path.join(ROOT_DIR, "outputs/models/efficientnet_b0_finetuned_best.pth"))
        print("✓ Best model saved.")

# =====================================
# STAGE 2: PROGRESSIVE UNFREEZING (last 2 blocks + head)
# =====================================

print("\n" + "="*60)
print("STAGE 2: Progressive Fine-tuning (Last 2 Blocks + Head)")
print("="*60)

# Progressive unfreeze: only last 2 blocks for lightweight model
# (more conservative than unfreezing blocks 6-8)
for name, param in model.named_parameters():
    if ("features.7" in name or "features.8" in name or
            "classifier" in name):
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=STAGE_2_EPOCHS)

for epoch in range(STAGE_2_EPOCHS):
    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE_2_EPOCHS}] - STAGE 2")

    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc   = 100 * train_correct / train_total
    train_loss /= len(train_loader)
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

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
        'epoch': global_epoch, 'stage': 'Stage 2',
        'train_loss': train_loss, 'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = global_epoch
        best_stage   = "Stage 2"
        torch.save(model.state_dict(),
                   os.path.join(ROOT_DIR, "outputs/models/best_efficientnet_b0.pth"))
        print("✓ Best model saved.")

print("\n" + "="*60)
print("Training Complete.")
print("="*60)
print(f"\nBest Val Accuracy: {best_val_acc:.2f}% at Epoch {best_epoch} ({best_stage})")

pd.DataFrame(metrics).to_csv(
    os.path.join(ROOT_DIR, "outputs/logs/efficientnet_b0_finetuned_metrics.csv"),
    index=False
)
print("Metrics saved.")