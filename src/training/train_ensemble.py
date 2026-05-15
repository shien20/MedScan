import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform
from src.models.ensemble import HeavyEnsemble, LightEnsemble

# =====================================
# CONFIG — change this to switch models
# =====================================

ENSEMBLE_TYPE = "heavy"   # "heavy" or "light"

ROOT_DIR = os.getcwd()

RESNET_PATH      = os.path.join(ROOT_DIR, "outputs/models/resnet50_baseline_best.pth")
DENSENET_PATH    = os.path.join(ROOT_DIR, "outputs/models/densenet121_baseline_best.pth")
EFFICIENTNET_PATH = os.path.join(ROOT_DIR, "outputs/models/efficientnet_b0_baseline_best.pth")
MOBILENET_PATH   = os.path.join(ROOT_DIR, "outputs/models/mobilenetv2_baseline_best.pth")

STAGE1_EPOCHS = 5     # frozen backbone, train head only
STAGE2_EPOCHS = 15    # unfreeze top layers, fine-tune everything
BATCH_SIZE = 16

os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)

# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =====================================
# DATASETS
# =====================================

train_csv = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv   = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

train_dataset = ChestXrayDataset(train_csv, images_dir, train_transform)
val_dataset   = ChestXrayDataset(val_csv,   images_dir, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# =====================================
# BUILD MODEL
# =====================================

if ENSEMBLE_TYPE == "heavy":
    model = HeavyEnsemble(
        resnet_path=RESNET_PATH,
        densenet_path=DENSENET_PATH
    )
    save_name = "best_heavy_ensemble.pth"
    log_name  = "heavy_ensemble_metrics.csv"
else:
    model = LightEnsemble(
        efficientnet_path=EFFICIENTNET_PATH,
        mobilenet_path=MOBILENET_PATH
    )
    save_name = "best_light_ensemble.pth"
    log_name  = "light_ensemble_metrics.csv"

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

metrics = []
best_val_acc = 0.0


# =====================================
# HELPER: one epoch of train + validate
# =====================================

def run_epoch(model, loader, optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    total_loss /= len(loader)  # Average loss per batch
    return total_loss, acc


def save_if_best(val_acc, model, path):
    global best_val_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), path)
        print(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")


model_save_path = os.path.join(ROOT_DIR, f"outputs/models/{save_name}")

# =====================================
# STAGE 1 — Frozen backbone
# Train only the classifier head
# =====================================

print("\n" + "="*50)
print("STAGE 1: Training classifier head (backbone frozen)")
print("="*50)

model.freeze_backbones()

optimizer_s1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

for epoch in range(STAGE1_EPOCHS):
    print(f"\nStage 1 — Epoch [{epoch+1}/{STAGE1_EPOCHS}]")

    train_loss, train_acc = run_epoch(model, train_loader, optimizer_s1, is_train=True)
    _, val_acc            = run_epoch(model, val_loader,   is_train=False)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")

    metrics.append({
        "stage": 1,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })

    save_if_best(val_acc, model, model_save_path)


# =====================================
# STAGE 2 — Unfreeze top layers
# Fine-tune with lower learning rate
# =====================================

print("\n" + "="*50)
print("STAGE 2: Fine-tuning top layers (backbone partially unfrozen)")
print("="*50)

model.unfreeze_top_layers()

optimizer_s2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

for epoch in range(STAGE2_EPOCHS):
    print(f"\nStage 2 — Epoch [{epoch+1}/{STAGE2_EPOCHS}]")

    train_loss, train_acc = run_epoch(model, train_loader, optimizer_s2, is_train=True)
    _, val_acc            = run_epoch(model, val_loader,   is_train=False)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")

    metrics.append({
        "stage": 2,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc
    })

    save_if_best(val_acc, model, model_save_path)


# =====================================
# SAVE METRICS
# =====================================

metrics_df = pd.DataFrame(metrics)
log_path = os.path.join(ROOT_DIR, f"outputs/logs/{log_name}")
metrics_df.to_csv(log_path, index=False)

print(f"\nTraining complete. Metrics saved to: {log_path}")
print(f"Best val accuracy: {best_val_acc:.2f}%")