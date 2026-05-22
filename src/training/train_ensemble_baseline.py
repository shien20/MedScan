"""
Simplified Ensemble Training (Baseline Models Only)
- Skips 2-stage training
- Loads baseline checkpoints directly
- Trains fusion classifier with frozen backbones (single unified training)
- Better performance since baseline models outperform finetuned versions
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import train_transform, val_transform
from src.models.ensemble import HeavyEnsemble, LightEnsemble

# =====================================
# CONFIG
# =====================================

ENSEMBLE_TYPE  = "heavy"   # change to "light" for lightweight ensemble
EPOCHS         = 20        # unified single training stage
BATCH_SIZE     = 16
PATIENCE       = 5         # early stopping patience
LEARNING_RATE  = 3e-4

ROOT_DIR = os.getcwd()

# Load baseline models only
RESNET_PATH      = os.path.join(ROOT_DIR, "outputs/models/baseline_resnet50.pth")
DENSENET_PATH    = os.path.join(ROOT_DIR, "outputs/models/baseline_densenet121.pth")
EFFICIENTNET_PATH = os.path.join(ROOT_DIR, "outputs/models/baseline_efficientnet_b0.pth")
MOBILENET_PATH   = os.path.join(ROOT_DIR, "outputs/models/baseline_mobilenetv2.pth")

os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"),   exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training: {ENSEMBLE_TYPE.upper()} ENSEMBLE (BASELINE MODELS)")
print(f"Training strategy: Single-stage (backbones frozen, classifier trained)")

# =====================================
# DATASETS
# =====================================

train_csv  = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv    = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

train_dataset = ChestXrayDataset(train_csv, images_dir, train_transform)
val_dataset   = ChestXrayDataset(val_csv,   images_dir, val_transform)

# =====================================
# WEIGHTED SAMPLER
# =====================================

def make_weighted_sampler(dataset):
    # Map classification strings to numeric labels
    label_map = {
        "Normal": 0,
        "Pneumonia": 1,
        "COVID-19": 2,
        "Tuberculosis": 3
    }
    
    classifications = dataset.df["classification"].values
    labels = torch.tensor([label_map[c] for c in classifications], dtype=torch.long)
    
    class_counts  = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=make_weighted_sampler(train_dataset),
    shuffle=False
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================================
# BUILD MODEL
# =====================================

if ENSEMBLE_TYPE == "heavy":
    model = HeavyEnsemble(
        resnet_path=RESNET_PATH,
        densenet_path=DENSENET_PATH
    )
    save_name = "baseline_heavy_ensemble.pth"
    log_name  = "baseline_heavy_ensemble_metrics.csv"
else:
    model = LightEnsemble(
        efficientnet_path=EFFICIENTNET_PATH,
        mobilenet_path=MOBILENET_PATH
    )
    save_name = "baseline_light_ensemble.pth"
    log_name  = "baseline_light_ensemble_metrics.csv"

model = model.to(device)

# =====================================
# FREEZE BACKBONES (keep them frozen throughout)
# =====================================

print("\nFreezing backbone layers...")
model.freeze_backbones()

# Verify only classifier is trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params     = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

# =====================================
# WEIGHTED LOSS
# =====================================

train_counts  = torch.tensor([12460, 4494, 2893, 969], dtype=torch.float)
class_weights = train_counts.sum() / (4 * train_counts)
class_weights = (class_weights / class_weights.sum() * 4).to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

# =====================================
# EARLY STOPPING HELPER
# =====================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.stop       = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True

# =====================================
# TRAINING HELPER
# =====================================

def run_epoch(model, loader, optimizer=None, is_train=True):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# =====================================
# TRAINING STATE
# =====================================

model_save_path = os.path.join(ROOT_DIR, f"outputs/models/{save_name}")
best_val_acc    = 0.0
best_val_loss   = float("inf")
best_epoch      = 0
metrics         = []
early_stopper   = EarlyStopping(patience=PATIENCE)

# =====================================
# UNIFIED TRAINING — Single Stage
# Train fusion classifier with frozen backbones
# =====================================

print("\n" + "="*60)
print("BASELINE ENSEMBLE TRAINING (Unified Stage)")
print("Backbones: FROZEN | Training: Fusion Classifier Only")
print("="*60)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    train_loss, train_acc = run_epoch(model, train_loader, optimizer, is_train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   is_train=False)

    scheduler.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")

    metrics.append({
        "epoch": epoch + 1,
        "stage": "Baseline",
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Best model tracking
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        torch.save(model.state_dict(), model_save_path)
        print(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")

    # Early stopping
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"\n✓ Early stopping triggered at epoch {epoch+1}")
        break

# =====================================
# SAVE METRICS & SUMMARY
# =====================================

metrics_df = pd.DataFrame(metrics)
metrics_csv_path = os.path.join(ROOT_DIR, f"outputs/logs/{log_name}")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\n✓ Metrics saved to: {metrics_csv_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Best Model:    {model_save_path}")
print(f"Best Val Acc:  {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"Best Val Loss: {best_val_loss:.4f}")
print(f"Metrics CSV:   {metrics_csv_path}")
print("="*60)
