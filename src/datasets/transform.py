from torchvision import transforms

# ─────────────────────────────────────────────────────────────────────────────
# IMAGENET NORMALISATION STATS (same for all transforms)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# MAJORITY CLASS TRANSFORM (Normal)
# Light augmentation — class is well represented
# ─────────────────────────────────────────────────────────────────────────────
majority_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ─────────────────────────────────────────────────────────────────────────────
# MINORITY CLASS TRANSFORM (Pneumonia, COVID-19, Tuberculosis)
# Stronger augmentation — simulate more visual diversity to help model
# generalise from fewer real samples
# ─────────────────────────────────────────────────────────────────────────────
minority_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),      # slight positional shift
        scale=(0.95, 1.05)           # slight zoom variation
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION / TEST TRANSFORM
# No augmentation — only resize and normalise (as required by your lecturer)
# ─────────────────────────────────────────────────────────────────────────────
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Keep train_transform as an alias for majority
# (used in places that don't need class-aware augmentation)
train_transform = majority_transform