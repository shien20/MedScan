import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import val_transform
from src.models.ensemble import HeavyEnsemble, LightEnsemble


# =====================================
# GET PROJECT ROOT DIRECTORY
# =====================================

ROOT_DIR = os.getcwd()

print(f"Project root: {ROOT_DIR}")


# =====================================
# DEVICE CONFIGURATION
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# =====================================
# CREATE OUTPUT DIRECTORIES
# =====================================

os.makedirs(os.path.join(ROOT_DIR, "outputs/confusion_matrix"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "outputs/logs"), exist_ok=True)


# =====================================
# LOAD TEST DATASET
# =====================================

test_csv_path = os.path.join(ROOT_DIR, "data/processed/all_data/test.csv")
images_dir_path = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

print(f"\nLoading test data from: {test_csv_path}")

test_dataset = ChestXrayDataset(
    csv_file=test_csv_path,
    image_dir=images_dir_path,
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

print(f"Test dataset size: {len(test_dataset)}")


# =====================================
# LOAD TRAINED ENSEMBLE MODEL
# =====================================

ENSEMBLE_TYPE = "heavy"   # change to "light" for the other ensemble

if ENSEMBLE_TYPE == "heavy":
    model = HeavyEnsemble()
    model_path = os.path.join(ROOT_DIR, "outputs/models/best_heavy_ensemble.pth")
    cm_save_name = "confusion_matrix_heavy_ensemble.png"
    report_prefix = "eval_heavy_ensemble"
else:
    model = LightEnsemble()
    model_path = os.path.join(ROOT_DIR, "outputs/models/best_light_ensemble.pth")
    cm_save_name = "confusion_matrix_light_ensemble.png"
    report_prefix = "eval_light_ensemble"

print(f"\nLoading model from: {model_path}")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
else:
    print(f"ERROR: Model file not found at {model_path}")
    exit()

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
plt.title(f'Confusion Matrix - {ENSEMBLE_TYPE.capitalize()} Ensemble Test Set', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()

# Save Confusion Matrix
cm_save_path = os.path.join(ROOT_DIR, f"outputs/confusion_matrix/{cm_save_name}")
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
print(f"\nConfusion Matrix saved to: {cm_save_path}")
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
eval_csv_path = os.path.join(ROOT_DIR, f"outputs/logs/evaluation_results_{report_prefix}.csv")
eval_df.to_csv(eval_csv_path, index=False)

print(f"\nEvaluation results saved to: {eval_csv_path}")

# Save detailed classification report to text file
report_txt_path = os.path.join(ROOT_DIR, f"outputs/logs/classification_report_{report_prefix}.txt")
with open(report_txt_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write(f"EVALUATION RESULTS - {ENSEMBLE_TYPE.upper()} ENSEMBLE\n")
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

print(f"Classification report saved to: {report_txt_path}")


# =====================================
# SUMMARY
# =====================================

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"\nGenerated outputs:")
print(f"  - Confusion Matrix: {cm_save_path}")
print(f"  - Evaluation Results: {eval_csv_path}")
print(f"  - Classification Report: {report_txt_path}")
print("\nUse these results for your thesis!")