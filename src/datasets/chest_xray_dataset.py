import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from src.datasets.transform import val_transform


CLASS_MAP = {
    "Normal": 0,
    "Pneumonia": 1,
    "COVID-19": 2,
    "Tuberculosis": 3
}


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, is_train=False):
        """
        Args:
            csv_file  : path to train.csv / val.csv / test.csv
            image_dir : path to Images folder
            transform : transform to apply to images
            is_train  : kept for backward compatibility, not used in final experiments
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform if transform is not None else val_transform

        if "label" not in self.df.columns:
            self.df["label"] = self.df["classification"].map(CLASS_MAP)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])

        img_path = os.path.join(self.image_dir, row["file_name"])
        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        return image, label