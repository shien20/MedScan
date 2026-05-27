import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from src.datasets.transform import majority_transform, minority_transform, val_transform


# Label index for each class
CLASS_MAP = {
    "Normal":       0,
    "Pneumonia":    1,
    "COVID-19":     2,
    "Tuberculosis": 3
}

# Which label indices are minority classes
# Normal (0) gets majority transform, everything else gets minority
MINORITY_CLASSES = {1, 2, 3}   # Pneumonia, COVID-19, Tuberculosis


class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None, is_train=False):
        """
        Args:
            csv_file  : path to train.csv / val.csv / test.csv
            image_dir : path to the Images folder
            transform : override transform (used for val/test)
            is_train  : if True, applies class-aware augmentation
                        if False, applies val_transform only
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.is_train = is_train
        self.transform = transform  # override, used for val/test

        # Label encoding
        self.label_map = CLASS_MAP

        # Map string labels to integers if not already done
        if "label" not in self.df.columns:
            self.df["label"] = self.df["classification"].map(CLASS_MAP)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row
        row = self.df.iloc[idx]
        label = int(row["label"])

        # Image path
        img_path = os.path.join(
            self.image_dir,
            row["file_name"]
        )

        # Load image
        image = Image.open(img_path).convert("RGB")

        # ── Transform selection ───────────────────────────────────────
        if self.transform is not None:
            # Explicit override — used for val/test loaders
            image = self.transform(image)

        elif self.is_train:
            # Class-aware augmentation — training only
            if label in MINORITY_CLASSES:
                image = minority_transform(image)   # stronger augmentation
            else:
                image = majority_transform(image)   # lighter augmentation

        else:
            # Default fallback — val/test behaviour
            image = val_transform(image)

        return image, label