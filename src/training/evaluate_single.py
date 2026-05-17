import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.nn.functional import softmax
from torchvision import models
from torch.utils.data import DataLoader

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import val_transform


# =====================================
# CONFIGURATION
# =====================================

# Usage: python evaluate_single.py MODEL_NAME MODEL_TYPE
# Example: python evaluate_single.py resnet50 baseline
# Example: python evaluate_single.py densenet121 finetuned

if len(sys.argv) < 3:
    print("Usage: python evaluate_single.py <model_name> <model_type>")
    print("  model_name: resnet50, densenet121, efficientnet_b0, mobilenetv2")
    print("  model_type: baseline, finetuned")
    sys.exit(1)

MODEL_NAME = sys.argv[1].lower()  # e.g., "resnet50", "densenet121"
MODEL_TYPE = sys.argv[2].lower()  # e.g., "baseline", "finetuned"

# Validate
if MODEL_TYPE not in ["baseline", "finetuned"]:
    print(f"ERROR: model_type must be 'baseline' or 'finetuned', got '{MODEL_TYPE}'")
    sys.exit(1)

if MODEL_NAME not in ["resnet50", "densenet121", "efficientnet_b0", "mobilenetv2"]:
    print(f"ERROR: model_name must be one of: resnet50, densenet121, efficientnet_b0, mobilenetv2")
    sys.exit(1)

ROOT_DIR = os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Project root: {ROOT_DIR}")
print(f"Using device: {device}")
print(f"Evaluating: {MODEL_NAME} ({MODEL_TYPE})")

os.makedirs(os.path.join(ROOT_DIR, "outputs/confusion_matrix"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)


# =====================================
# LOAD TEST DATASET
# =====================================

test_csv_path = os.path.join(ROOT_DIR, "data/processed/all_data/test.csv")
images_dir_path = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

test_dataset = ChestXrayDataset(csv_file=test_csv_path, image_dir=images_dir_path, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Test dataset size: {len(test_dataset)}")


# =====================================
# CREATE MODEL ARCHITECTURE
# =====================================

if MODEL_NAME == "resnet50":
    model = models.resnet50(weights="IMAGENET1K_V1")
    if MODEL_TYPE == "baseline":
        model.fc = nn.Linear(model.fc.in_features, 4)
    else:  # finetuned
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
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

elif MODEL_NAME == "densenet121":
    model = models.densenet121(weights="IMAGENET1K_V1")
    if MODEL_TYPE == "baseline":
        model.classifier = nn.Linear(model.classifier.in_features, 4)
    else:  # finetuned
        in_feat = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4)
        )

elif MODEL_NAME == "efficientnet_b0":
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    if MODEL_TYPE == "baseline":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    else:  # finetuned
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_feat, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )

elif MODEL_NAME == "mobilenetv2":
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    if MODEL_TYPE == "baseline":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    else:  # finetuned
        in_feat = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_feat, 1024),
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
# LOAD TRAINED MODEL WEIGHTS
# =====================================

# Construct model path based on model type
if MODEL_TYPE == "baseline":
    model_path = os.path.join(ROOT_DIR, f"outputs/models/baseline_{MODEL_NAME}.pth")
else:  # finetuned
    model_path = os.path.join(ROOT_DIR, f"outputs/models/best_{MODEL_NAME}.pth")

print(f"Loading model from: {model_path}")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✓ Model loaded successfully!")
else:
    print(f"ERROR: Model file not found at {model_path}")
    sys.exit(1)

model = model.to(device)
model.eval()


# =====================================
# PREDICTIONS ON TEST SET
# =====================================

print("\nRunning predictions on test set...")

all_predictions = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# =====================================
# CLASS LABELS
# =====================================

class_names = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]
class_map = {0: "Normal", 1: "Pneumonia", 2: "COVID-19", 3: "Tuberculosis"}


# =====================================
# EVALUATION METRICS
# =====================================

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)

# Overall Accuracy
test_accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")

# Per-class Recall (IMPORTANT FOR MEDICAL AI)
recall_per_class = recall_score(all_labels, all_predictions, average=None)
print("\n--- Recall per Class (Important for Medical AI) ---")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {recall_per_class[i]*100:.2f}%")

# Per-class F1-Score
f1_per_class = f1_score(all_labels, all_predictions, average=None)
print("\n--- F1-Score per Class ---")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {f1_per_class[i]*100:.2f}%")

# AUC-ROC (one-vs-rest, macro average)
print("\n--- AUC-ROC (One-vs-Rest, Macro Average) ---")
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
auc_score = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr', average='macro')
print(f"Macro AUC-ROC: {auc_score:.4f}")

# Per-class AUC-ROC
print("\n--- Per-Class AUC-ROC ---")
per_class_auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr', average=None)
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {per_class_auc[i]:.4f}")

# Detailed Classification Report
print("\n--- Detailed Classification Report ---")
class_report = classification_report(
    all_labels, 
    all_predictions, 
    target_names=class_names,
    digits=4
)
print(class_report)


# =====================================
# CONFUSION MATRIX
# =====================================

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_predictions)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()

# Save Confusion Matrix with proper naming
cm_save_path = os.path.join(ROOT_DIR, f"outputs/confusion_matrix/confusion_matrix_{MODEL_NAME}_{MODEL_TYPE}.png")
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion Matrix saved to: {cm_save_path}")
plt.close()


# =====================================
# SAVE EVALUATION RESULTS TO CSV
# =====================================

# Create evaluation summary
evaluation_results = {
    'Metric': [
        'Test Accuracy',
        'Normal - Recall',
        'Pneumonia - Recall',
        'COVID-19 - Recall',
        'Tuberculosis - Recall',
        'Normal - F1-Score',
        'Pneumonia - F1-Score',
        'COVID-19 - F1-Score',
        'Tuberculosis - F1-Score',
        'Macro AUC-ROC',
        'Normal - AUC-ROC',
        'Pneumonia - AUC-ROC',
        'COVID-19 - AUC-ROC',
        'Tuberculosis - AUC-ROC'
    ],
    'Value': [
        test_accuracy,
        recall_per_class[0],
        recall_per_class[1],
        recall_per_class[2],
        recall_per_class[3],
        f1_per_class[0],
        f1_per_class[1],
        f1_per_class[2],
        f1_per_class[3],
        auc_score,
        per_class_auc[0],
        per_class_auc[1],
        per_class_auc[2],
        per_class_auc[3]
    ]
}

eval_df = pd.DataFrame(evaluation_results)
eval_csv_path = os.path.join(ROOT_DIR, f"outputs/logs/evaluation_results_{MODEL_NAME}_{MODEL_TYPE}.csv")
eval_df.to_csv(eval_csv_path, index=False)

print(f"\n✓ Evaluation results saved to: {eval_csv_path}")

# Save detailed classification report to text file with proper naming
report_txt_path = os.path.join(ROOT_DIR, f"outputs/logs/classification_report_{MODEL_NAME}_{MODEL_TYPE}.txt")
with open(report_txt_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write(f"EVALUATION RESULTS - {MODEL_NAME.upper()} ({MODEL_TYPE.upper()})\n")
    f.write("="*60 + "\n\n")
    f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n\n")
    f.write("--- Recall per Class ---\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {recall_per_class[i]*100:.2f}%\n")
    f.write("\n--- F1-Score per Class ---\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {f1_per_class[i]*100:.2f}%\n")
    f.write("\n--- AUC-ROC Scores ---\n")
    f.write(f"Macro AUC-ROC: {auc_score:.4f}\n")
    f.write("Per-Class AUC-ROC:\n")
    for i, class_name in enumerate(class_names):
        f.write(f"{class_name}: {per_class_auc[i]:.4f}\n")
    f.write("\n--- Classification Report ---\n")
    f.write(class_report)
    f.write("\n--- Confusion Matrix ---\n")
    f.write(str(cm))

print(f"✓ Classification report saved to: {report_txt_path}")


# =====================================
# SUMMARY
# =====================================

print("\n" + "="*60)
print(f"✓ EVALUATION COMPLETE - {MODEL_NAME.upper()} ({MODEL_TYPE.upper()})")
print("="*60)
print(f"\nGenerated outputs:")
print(f"  - Confusion Matrix: confusion_matrix_{MODEL_NAME}_{MODEL_TYPE}.png")
print(f"  - Evaluation Results: evaluation_results_{MODEL_NAME}_{MODEL_TYPE}.csv")
print(f"  - Classification Report: classification_report_{MODEL_NAME}_{MODEL_TYPE}.txt")
print(f"\nAll files saved to:")
print(f"  - {os.path.join(ROOT_DIR, 'outputs/confusion_matrix/')}")
print(f"  - {os.path.join(ROOT_DIR, 'outputs/logs/')}")
