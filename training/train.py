import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from cnn_model import CropDiseaseModel



BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train"
MODEL_DIR = BASE_DIR / "model"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
#dataset loading

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

print(f"Total Images : {len(dataset)}")
print(f"Total Classes : {len(dataset.classes)}")
print(dataset.classes)

num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



model = CropDiseaseModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_acc = 0
best_weights = copy.deepcopy(model.state_dict())

train_acc_history = []
val_acc_history = []


for epoch in range(EPOCHS):

    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

    model.train()

    correct = 0
    total = 0
    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}")

    train_acc = 100 * correct / total

  #validation

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Train Accuracy : {train_acc:.2f}%")
    print(f"Validation Accuracy : {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())


MODEL_DIR.mkdir(exist_ok=True)

model.load_state_dict(best_weights)

torch.save(
    model.state_dict(),
    MODEL_DIR / "crop_model.pth"
)

print("\nModel Saved Successfully!")

#graph

plt.figure(figsize=(8,5))

plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(val_acc_history, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(MODEL_DIR / "accuracy.png")

plt.show()