import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_dir, transform=None):

        self.df = pd.read_csv(csv_file)

        self.image_dir = image_dir
        self.transform = transform

        # Label encoding
        self.label_map = {
            "Normal": 0,
            "Pneumonia": 1,
            "COVID-19": 2,
            "Tuberculosis": 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get row
        row = self.df.iloc[idx]

        # Image path
        img_path = os.path.join(
            self.image_dir,
            row["file_name"]
        )

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get label
        label = self.label_map[row["classification"]]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label