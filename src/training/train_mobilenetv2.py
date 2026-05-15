import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import models
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform

# =====================================
# GET PROJECT ROOT DIRECTORY
# =====================================

ROOT_DIR = os.getcwd()

# Create output directories if they don't exist
os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/confusion_matrix"), exist_ok=True)


# =====================================
# DEVICE CONFIGURATION
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# =====================================
# DATASETS
# =====================================

train_csv_path = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv_path = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir_path = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

# Debug: Print paths
print(f"\nProject root: {ROOT_DIR}")
print(f"Train CSV path: {train_csv_path}")
print(f"Val CSV path: {val_csv_path}")
print(f"Images dir path: {images_dir_path}")
print(f"Train CSV exists: {os.path.exists(train_csv_path)}")
print(f"Images dir exists: {os.path.exists(images_dir_path)}\n")

train_dataset = ChestXrayDataset(
    csv_file=train_csv_path,
    image_dir=images_dir_path,
    transform=train_transform
)

val_dataset = ChestXrayDataset(
    csv_file=val_csv_path,
    image_dir=images_dir_path,
    transform=val_transform
)


# =====================================
# DATALOADERS
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# =====================================
# LOAD PRETRAINED MOBILENETV2
# =====================================

model = models.mobilenet_v2(weights="IMAGENET1K_V1")


# =====================================
# REPLACE FINAL LAYER
# =====================================

num_features = model.classifier[1].in_features  # 1280 for MobileNetV2

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_features, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)


# =====================================
# MOVE MODEL TO GPU
# =====================================

model = model.to(device)


# =====================================
# LOSS FUNCTION (with label smoothing)
# =====================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# =====================================
# TRAINING SETTINGS
# =====================================

STAGE_1_EPOCHS = 5  # Freeze backbone, train head only
STAGE_2_EPOCHS = 10  # Fine-tune top layers + head

best_val_acc = 0.0
best_epoch = 0
best_stage = ""

# Initialize metrics list to track training progress
metrics = []


# =====================================
# STAGE 1: FREEZE BACKBONE, TRAIN HEAD ONLY (5 epochs)
# =====================================

print("\n" + "="*60)
print("STAGE 1: Training Classification Head (Backbone Frozen)")
print("="*60)

# Freeze backbone layers (everything except classifier)
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

# Create optimizer for Stage 1 (only trainable parameters)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3  # Higher learning rate for stage 1
)

# Learning rate scheduler for Stage 1
scheduler = CosineAnnealingLR(optimizer, T_max=STAGE_1_EPOCHS)

global_epoch = 0

for epoch in range(STAGE_1_EPOCHS):

    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE_1_EPOCHS}] - STAGE 1")

    # =========================
    # TRAINING MODE
    # =========================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader):

        # Move to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    train_loss /= len(train_loader)  # Average loss per batch

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    # Step the scheduler
    scheduler.step()


    # =========================
    # VALIDATION MODE
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Validation Accuracy: {val_acc:.2f}%")

    # =========================
    # SAVE METRICS FOR THIS EPOCH
    # =========================

    metrics.append({
        'epoch': global_epoch,
        'stage': 'Stage 1',
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = global_epoch
        best_stage = "Stage 1"

        model_save_path = os.path.join(ROOT_DIR, "outputs/models/best_mobilenetv2.pth")
        
        torch.save(
            model.state_dict(),
            model_save_path
        )

        print("✓ Best model saved.")


# =====================================
# STAGE 2: UNFREEZE TOP LAYERS, FINE-TUNE (10 epochs)
# =====================================

print("\n" + "="*60)
print("STAGE 2: Fine-tuning Top Layers + Head")
print("="*60)

# Unfreeze top inverted residual blocks and classifier
for name, param in model.named_parameters():
    if "features.15" in name or "features.16" in name or "features.17" in name or "features.18" in name or "classifier" in name:
        param.requires_grad = True

# Create optimizer for Stage 2 (all trainable parameters now)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4  # Lower learning rate for fine-tuning
)

# Learning rate scheduler for Stage 2
scheduler = CosineAnnealingLR(optimizer, T_max=STAGE_2_EPOCHS)

for epoch in range(STAGE_2_EPOCHS):

    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE_2_EPOCHS}] - STAGE 2")

    # =========================
    # TRAINING MODE
    # =========================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader):

        # Move to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total
    train_loss /= len(train_loader)  # Average loss per batch

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    # Step the scheduler
    scheduler.step()


    # =========================
    # VALIDATION MODE
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Validation Accuracy: {val_acc:.2f}%")

    # =========================
    # SAVE METRICS FOR THIS EPOCH
    # =========================

    metrics.append({
        'epoch': global_epoch,
        'stage': 'Stage 2',
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })

    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = global_epoch
        best_stage = "Stage 2"

        model_save_path = os.path.join(ROOT_DIR, "outputs/models/best_mobilenetv2.pth")
        
        torch.save(
            model.state_dict(),
            model_save_path
        )

        print("✓ Best model saved.")


print("\n" + "="*60)
print("Training Complete.")
print("="*60)
print(f"\nBest Model Performance:")
print(f"  Validation Accuracy: {best_val_acc:.2f}%")
print(f"  At Epoch: {best_epoch}")
print(f"  Stage: {best_stage}")

# =====================================
# SAVE TRAINING METRICS
# =====================================

# Convert metrics list to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_csv_path = os.path.join(ROOT_DIR, "outputs/logs/training_metrics_mobilenetv2.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"\nTraining metrics saved to: {metrics_csv_path}")
print("\nMetrics Summary:")
print(metrics_df)
