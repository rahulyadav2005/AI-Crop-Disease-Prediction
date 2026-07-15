import os
import copy
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from cnn_model import CropDiseaseModel

# ==========================
# SETTINGS
# ==========================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train"

IMAGE_SIZE = 224

BATCH_SIZE = 32 

EPOCHS = 10

LEARNING_RATE = 0.001

# ==========================
# DEVICE
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

# ==========================
# TRANSFORM
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
# LOAD DATASET
# ==========================

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

num_classes = len(dataset.classes)

print("\nClasses:", dataset.classes)

# 80% Train | 20% Validation

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# MODEL
# ==========================

model = CropDiseaseModel(num_classes)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ==========================
# HISTORY
# ==========================

train_loss_history = []
val_loss_history = []

train_acc_history = []
val_acc_history = []

best_accuracy = 0
best_weights = copy.deepcopy(model.state_dict())

# ==========================
# TRAINING
# ==========================

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # -------- Train --------

    model.train()

    train_loss = 0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -------- Validation --------

    model.eval()

    val_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    train_loss_history.append(train_loss)

    val_loss_history.append(val_loss)

    train_acc_history.append(train_accuracy)

    val_acc_history.append(val_accuracy)

    print(f"Train Accuracy : {train_accuracy:.2f}%")
    print(f"Validation Accuracy : {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(model.state_dict())

# ==========================
# SAVE MODEL
# ==========================

os.makedirs("../model", exist_ok=True)

model.load_state_dict(best_weights)

torch.save(
    model.state_dict(),
    "../model/crop_model.pth"
)

print("\nBest Model Saved Successfully!")

# ==========================
# GRAPH
# ==========================

plt.figure(figsize=(8,5))

plt.plot(train_acc_history, label="Train Accuracy")

plt.plot(val_acc_history, label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("../model/accuracy.png")

plt.show()