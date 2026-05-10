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
# LOAD PRETRAINED RESNET50
# =====================================

model = models.resnet50(weights="IMAGENET1K_V1")


# =====================================
# REPLACE FINAL LAYER
# =====================================

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 4)


# =====================================
# MOVE MODEL TO GPU
# =====================================

model = model.to(device)


# =====================================
# LOSS FUNCTION
# =====================================

criterion = nn.CrossEntropyLoss()


# =====================================
# OPTIMIZER
# =====================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# =====================================
# TRAINING SETTINGS
# =====================================

num_epochs = 10

best_val_acc = 0.0

# Initialize metrics list to track training progress
metrics = []


# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(num_epochs):

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")

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

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")


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
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc
    })


    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        model_save_path = os.path.join(ROOT_DIR, "outputs/models/best_resnet50.pth")
        
        torch.save(
            model.state_dict(),
            model_save_path
        )

        print("Best model saved.")


print("\nTraining Complete.")

# =====================================
# SAVE TRAINING METRICS
# =====================================

# Convert metrics list to DataFrame
metrics_df = pd.DataFrame(metrics)

# Save to CSV
metrics_csv_path = os.path.join(ROOT_DIR, "outputs/logs/training_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"\nTraining metrics saved to: {metrics_csv_path}")
print("\nMetrics Summary:")
print(metrics_df)