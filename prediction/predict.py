
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "train"

print("Dataset Path:", DATASET_PATH)
print("Path Exists:", DATASET_PATH.exists())

import torch
from PIL import Image
from torchvision import transforms

from training.cnn_model import CropDiseaseModel

# ----------------------------
# Class Names
# ----------------------------

CLASS_NAMES = [
    "Corn Common Rust",
    "Corn Healthy",
    "Corn Northern Leaf Blight",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Early Blight",
    "Tomato Healthy",
    "Tomato Late Blight"
]

# ----------------------------
# Image Transform
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# Load Model
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CropDiseaseModel(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("model/crop_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Prediction Function
# ----------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return {
        "disease": CLASS_NAMES[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


# ----------------------------
# Test
# ----------------------------

if __name__ == "__main__":

    result = predict_image("test.jpg")

    print(result)