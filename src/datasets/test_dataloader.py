from torch.utils.data import DataLoader

from chest_xray_dataset import ChestXrayDataset
from transform import train_transform


# Create dataset
train_dataset = ChestXrayDataset(
    csv_file="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\train.csv",
    image_dir="C:\\Users\\shien\\OneDrive - Sunway Education Group\\Sunway materials\\Capstone Project 2\\MedScan\\data\\processed\\all_data\\Images",
    transform=train_transform
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# Test one batch
# Read metadata.csv and pick 16 random rows, then load the corresponding images and labels
# Apply image transformations and convert images to tensors 
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
# Image batch shape: torch.Size([16, 3, 224, 224])
# (number of images, color channels, image height, image width)
print("Labels:", labels)
# Labels: tensor([0, 0, 1, 2, 2, 0, 0, 3, 0, 1, 0, 0, 1, 0, 2, 0])
# disease categories that we have converted from text to numbers