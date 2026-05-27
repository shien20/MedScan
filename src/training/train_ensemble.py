import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import val_transform
from src.models.ensemble import HeavyEnsemble, LightEnsemble

# =====================================
# CONFIG — switch between "heavy" / "light"
# =====================================

ENSEMBLE_TYPE  = "heavy"   # change to "light" for lightweight ensemble
STAGE1_EPOCHS  = 5
STAGE2_EPOCHS  = 10
BATCH_SIZE     = 16
PATIENCE       = 5         # early stopping patience

ROOT_DIR = os.getcwd()

RESNET_PATH      = os.path.join(ROOT_DIR, "outputs/models/best_resnet50.pth")
DENSENET_PATH    = os.path.join(ROOT_DIR, "outputs/models/baseline_densenet121.pth")
EFFICIENTNET_PATH = os.path.join(ROOT_DIR, "outputs/models/baseline_efficientnet_b0.pth")
MOBILENET_PATH   = os.path.join(ROOT_DIR, "outputs/models/baseline_mobilenetv2.pth")

os.makedirs(os.path.join(ROOT_DIR, "outputs/models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"),   exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training: {ENSEMBLE_TYPE.upper()} ENSEMBLE")

# =====================================
# DATASETS
# =====================================

train_csv  = os.path.join(ROOT_DIR, "data/processed/all_data/train.csv")
val_csv    = os.path.join(ROOT_DIR, "data/processed/all_data/val.csv")
images_dir = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

# =====================================
# TRAIN DATASET - Class-aware augmentation
# =====================================
train_dataset = ChestXrayDataset(
    csv_file=train_csv,
    image_dir=images_dir,
    transform=None,      # Use class-aware logic inside dataset
    is_train=True        # Enable minority/majority split
)

print(f"\nClass distribution in training set:")
class_names = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]
for class_idx, class_name in enumerate(class_names):
    count = (train_dataset.df["label"] == class_idx).sum()
    print(f"  {class_name}: {count} images")

# =====================================
# VALIDATION DATASET - No augmentation
# =====================================
val_dataset = ChestXrayDataset(
    csv_file=val_csv,
    image_dir=images_dir,
    transform=val_transform,
    is_train=False
)

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
best_stage      = ""
metrics         = []
global_epoch    = 0
early_stopper   = EarlyStopping(patience=PATIENCE)

# =====================================
# STAGE 1 — Frozen backbones
# Train fusion classifier head only
# =====================================

print("\n" + "="*60)
print("STAGE 1: Training Fusion Classifier (Backbones Frozen)")
print("="*60)

model.freeze_backbones()

optimizer_s1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4
)
scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=STAGE1_EPOCHS)

for epoch in range(STAGE1_EPOCHS):
    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE1_EPOCHS}] — Stage 1")

    train_loss, train_acc = run_epoch(model, train_loader, optimizer_s1, is_train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   is_train=False)

    scheduler_s1.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")

    metrics.append({
        "epoch": global_epoch, "stage": "Stage 1",
        "train_loss": train_loss, "train_accuracy": train_acc,
        "val_loss": val_loss, "val_accuracy": val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        best_epoch    = global_epoch
        best_stage    = "Stage 1"
        torch.save(model.state_dict(), model_save_path)
        print(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")

# =====================================
# STAGE 2 — Unfreeze top layers
# Fine-tune with lower LR + early stopping
# =====================================

print("\n" + "="*60)
print("STAGE 2: Fine-Tuning Top Backbone Layers")
print("="*60)

model.unfreeze_top_layers()

# Slightly lower LR than individual model fine-tuning
# because we are updating two backbones simultaneously
optimizer_s2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-5
)
scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=STAGE2_EPOCHS)

for epoch in range(STAGE2_EPOCHS):
    global_epoch += 1
    print(f"\nEpoch [{epoch+1}/{STAGE2_EPOCHS}] — Stage 2")

    train_loss, train_acc = run_epoch(model, train_loader, optimizer_s2, is_train=True)
    val_loss,   val_acc   = run_epoch(model, val_loader,   is_train=False)

    scheduler_s2.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.2f}%")

    metrics.append({
        "epoch": global_epoch, "stage": "Stage 2",
        "train_loss": train_loss, "train_accuracy": train_acc,
        "val_loss": val_loss, "val_accuracy": val_acc
    })

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        best_epoch    = global_epoch
        best_stage    = "Stage 2"
        torch.save(model.state_dict(), model_save_path)
        print(f"  ✓ Best model saved (val acc: {val_acc:.2f}%)")

    # Early stopping based on validation loss
    early_stopper.step(val_loss)
    if early_stopper.stop:
        print(f"\n  Early stopping triggered at epoch {global_epoch}.")
        break

# =====================================
# SAVE METRICS
# =====================================

metrics_df = pd.DataFrame(metrics)
log_path   = os.path.join(ROOT_DIR, f"outputs/logs/{log_name}")
metrics_df.to_csv(log_path, index=False)

print("\n" + "="*60)
print("Training Complete.")
print("="*60)
print(f"Best Val Accuracy : {best_val_acc:.2f}%")
print(f"At Epoch          : {best_epoch} ({best_stage})")
print(f"Metrics saved to  : {log_path}")
print(f"Model saved to    : {model_save_path}")