from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

print("Total Images:", len(dataset))
print("Classes:", len(dataset.classes))

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print("Loading first batch...")

images, labels = next(iter(loader))

print("Image Shape:", images.shape)
print("Labels:", labels)

print("Dataset Working Successfully!")