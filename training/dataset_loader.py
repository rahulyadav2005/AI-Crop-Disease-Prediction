import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================
# Settings
# ==========================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train"

IMAGE_SIZE = 224

BATCH_SIZE = 32


# ==========================
# Image Transformation
# ==========================

transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])


# ==========================
# Load Dataset
# ==========================

dataset = datasets.ImageFolder(

    DATASET_PATH,

    transform=transform

)


# ==========================
# Class Names
# ==========================

class_names = dataset.classes


print("\nClasses Found:\n")

for name in class_names:
    print(name)


# ==========================
# Data Loader
# ==========================

data_loader = DataLoader(

    dataset,

    batch_size=BATCH_SIZE,

    shuffle=True

)


print("\nDataset Loaded Successfully")

print("Total Images:", len(dataset))