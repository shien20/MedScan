import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform


# =====================================
# DEVICE CONFIGURATION
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# =====================================
# DATASETS
# =====================================

train_dataset = ChestXrayDataset(
    csv_file="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\train.csv",
    image_dir="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\Images",
    transform=train_transform
)

val_dataset = ChestXrayDataset(
    csv_file="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\val.csv",
    image_dir="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\Images",
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
    # SAVE BEST MODEL
    # =========================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "outputs/models/best_resnet50.pth"
        )

        print("Best model saved.")


print("\nTraining Complete.")