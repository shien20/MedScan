import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.datasets.chest_xray_dataset import ChestXrayDataset
from src.datasets.transform import val_transform
from src.models.ensemble import HeavyEnsemble, LightEnsemble

# =====================================
# CONFIG
# =====================================

ENSEMBLE_TYPE = "heavy"   # change to "light" for lightweight ensemble

ROOT_DIR    = os.getcwd()
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19", "Tuberculosis"]

# Output directories
confusion_matrix_output_path = os.path.join(ROOT_DIR, "outputs/confusion_matrix")
logs_output_path             = os.path.join(ROOT_DIR, "outputs/logs")

os.makedirs(confusion_matrix_output_path, exist_ok=True)
os.makedirs(logs_output_path,             exist_ok=True)

print(f"Using device: {device}")
print(f"Evaluating: {ENSEMBLE_TYPE.upper()} ENSEMBLE")


# =====================================
# CONFUSION MATRIX FUNCTION
# =====================================

def plot_large_confusion_matrix(cm, title, filename,
                                class_names=["Normal", "Pneumonia",
                                             "COVID-19", "Tuberculosis"]):
    fig, ax = plt.subplots(figsize=(28, 26))

    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count", "shrink": 0.8},
        ax=ax,
        cbar=True,
        square=True,
        linewidths=4,
        linecolor="black"
    )

    # Manually add ultra-large annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=48, fontweight="bold",
                    color=text_color)

    ax.set_xlabel("Predicted Label", fontsize=48, fontweight="bold", labelpad=30)
    ax.set_ylabel("Actual Label",    fontsize=48, fontweight="bold", labelpad=30)
    ax.set_title(title, fontsize=54, fontweight="bold", pad=50)

    ax.set_xticklabels(class_names, fontsize=44, fontweight="bold",
                       rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=44, fontweight="bold",
                       rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=32)
    cbar.set_label("Count", fontsize=38, fontweight="bold")

    plt.tight_layout()

    save_path = os.path.join(confusion_matrix_output_path, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    print(f"  ✓ Saved confusion matrix: {filename}")

    plt.show()
    plt.close()


# =====================================
# LOAD TEST DATASET
# =====================================

test_csv   = os.path.join(ROOT_DIR, "data/processed/all_data/test.csv")
images_dir = os.path.join(ROOT_DIR, "data/processed/all_data/Images")

test_dataset = ChestXrayDataset(test_csv, images_dir, val_transform)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"\nTest dataset size: {len(test_dataset)}")


# =====================================
# LOAD MODEL
# =====================================

RESNET_PATH       = os.path.join(ROOT_DIR, "outputs/models/best_resnet50.pth")
DENSENET_PATH     = os.path.join(ROOT_DIR, "outputs/models/baseline_densenet121.pth")
EFFICIENTNET_PATH = os.path.join(ROOT_DIR, "outputs/models/baseline_efficientnet_b0.pth")
MOBILENET_PATH    = os.path.join(ROOT_DIR, "outputs/models/baseline_mobilenetv2.pth")

if ENSEMBLE_TYPE == "heavy":
    model          = HeavyEnsemble(RESNET_PATH, DENSENET_PATH)
    ensemble_path  = os.path.join(ROOT_DIR, "outputs/models/best_heavy_ensemble.pth")
    label          = "HEAVY_ENSEMBLE"
    cm_filename    = "confusion_matrix_heavy_ensemble.png"
    csv_filename   = "evaluation_heavy_ensemble.csv"
    report_filename = "classification_report_heavy_ensemble.txt"
else:
    model          = LightEnsemble(EFFICIENTNET_PATH, MOBILENET_PATH)
    ensemble_path  = os.path.join(ROOT_DIR, "outputs/models/best_light_ensemble.pth")
    label          = "LIGHT_ENSEMBLE"
    cm_filename    = "confusion_matrix_light_ensemble.png"
    csv_filename   = "evaluation_light_ensemble.csv"
    report_filename = "classification_report_light_ensemble.txt"

model.load_state_dict(torch.load(ensemble_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded: {ensemble_path}")


# =====================================
# INFERENCE
# =====================================

print("\nRunning inference on test set...")

all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(device)
        outputs = model(images)
        probs   = softmax(outputs, dim=1)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)


# =====================================
# COMPUTE METRICS
# =====================================

# Overall accuracy
overall_accuracy = accuracy_score(all_labels, all_preds)

# Per-class metrics
precision_per_class = precision_score(all_labels, all_preds, average=None)
recall_per_class    = recall_score(all_labels, all_preds, average=None)
f1_per_class        = f1_score(all_labels, all_preds, average=None)

# Macro averages
macro_precision = precision_score(all_labels, all_preds, average="macro")
macro_recall    = recall_score(all_labels, all_preds, average="macro")
macro_f1        = f1_score(all_labels, all_preds, average="macro")

# AUC-ROC
labels_bin      = label_binarize(all_labels, classes=[0, 1, 2, 3])
per_class_auc   = roc_auc_score(labels_bin, all_probs,
                                multi_class="ovr", average=None)
macro_auc       = roc_auc_score(labels_bin, all_probs,
                                multi_class="ovr", average="macro")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Detailed classification report
class_report = classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    digits=4
)


# =====================================
# PRINT RESULTS
# =====================================

print(f"\n{'='*60}")
print(f"EVALUATION RESULTS — {label}")
print(f"{'='*60}")

print(f"\nTest Accuracy: {overall_accuracy * 100:.2f}%")

print("\n--- Precision per Class ---")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {precision_per_class[i] * 100:.2f}%")
print(f"  Macro Precision: {macro_precision * 100:.2f}%")

print("\n--- Recall per Class ---")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {recall_per_class[i] * 100:.2f}%")
print(f"  Macro Recall: {macro_recall * 100:.2f}%")

print("\n--- F1-Score per Class ---")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {f1_per_class[i] * 100:.2f}%")
print(f"  Macro F1: {macro_f1 * 100:.2f}%")

print("\n--- AUC-ROC per Class ---")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: {per_class_auc[i]:.4f}")
print(f"  Macro AUC-ROC: {macro_auc:.4f}")

print("\n--- Classification Report ---")
print(class_report)

print("--- Confusion Matrix ---")
print(cm)


# =====================================
# PLOT CONFUSION MATRIX
# =====================================

plot_large_confusion_matrix(
    cm=cm,
    title=f"Confusion Matrix — {label.replace('_', ' ').title()}",
    filename=cm_filename
)


# =====================================
# SAVE RESULTS TO CSV
# =====================================

results_data = {
    "Metric": [
        # Accuracy
        "Test Accuracy",

        # Precision
        "Normal — Precision",
        "Pneumonia — Precision",
        "COVID-19 — Precision",
        "Tuberculosis — Precision",
        "Macro Precision",

        # Recall
        "Normal — Recall",
        "Pneumonia — Recall",
        "COVID-19 — Recall",
        "Tuberculosis — Recall",
        "Macro Recall",

        # F1
        "Normal — F1",
        "Pneumonia — F1",
        "COVID-19 — F1",
        "Tuberculosis — F1",
        "Macro F1",

        # AUC
        "Normal — AUC-ROC",
        "Pneumonia — AUC-ROC",
        "COVID-19 — AUC-ROC",
        "Tuberculosis — AUC-ROC",
        "Macro AUC-ROC"
    ],
    "Value": [
        overall_accuracy,

        precision_per_class[0], precision_per_class[1],
        precision_per_class[2], precision_per_class[3],
        macro_precision,

        recall_per_class[0], recall_per_class[1],
        recall_per_class[2], recall_per_class[3],
        macro_recall,

        f1_per_class[0], f1_per_class[1],
        f1_per_class[2], f1_per_class[3],
        macro_f1,

        per_class_auc[0], per_class_auc[1],
        per_class_auc[2], per_class_auc[3],
        macro_auc
    ]
}

csv_path = os.path.join(logs_output_path, csv_filename)
pd.DataFrame(results_data).to_csv(csv_path, index=False)
print(f"\nEvaluation results saved: {csv_path}")


# =====================================
# SAVE CLASSIFICATION REPORT TO TXT
# =====================================

report_path = os.path.join(logs_output_path, report_filename)

with open(report_path, "w") as f:
    f.write(f"{'='*60}\n")
    f.write(f"EVALUATION RESULTS — {label}\n")
    f.write(f"{'='*60}\n\n")

    f.write(f"Test Accuracy: {overall_accuracy * 100:.2f}%\n\n")

    f.write("--- Precision per Class ---\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {name}: {precision_per_class[i] * 100:.2f}%\n")
    f.write(f"  Macro Precision: {macro_precision * 100:.2f}%\n\n")

    f.write("--- Recall per Class ---\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {name}: {recall_per_class[i] * 100:.2f}%\n")
    f.write(f"  Macro Recall: {macro_recall * 100:.2f}%\n\n")

    f.write("--- F1-Score per Class ---\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {name}: {f1_per_class[i] * 100:.2f}%\n")
    f.write(f"  Macro F1: {macro_f1 * 100:.2f}%\n\n")

    f.write("--- AUC-ROC per Class ---\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"  {name}: {per_class_auc[i]:.4f}\n")
    f.write(f"  Macro AUC-ROC: {macro_auc:.4f}\n\n")

    f.write("--- Classification Report ---\n")
    f.write(class_report)
    f.write("\n--- Confusion Matrix ---\n")
    f.write(str(cm))

print(f"Classification report saved: {report_path}")

print(f"\n{'='*60}")
print("EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"  Confusion Matrix : outputs/confusion_matrix/{cm_filename}")
print(f"  CSV Results      : outputs/logs/{csv_filename}")
print(f"  Text Report      : outputs/logs/{report_filename}")