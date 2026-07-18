import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

#project path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.cnn_model import CropDiseaseModel

#setting

MODEL_PATH = BASE_DIR / "model" / "crop_model.pth"

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loading

model = CropDiseaseModel(len(CLASS_NAMES))

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)

model.eval()

#image transform

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(output, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    disease = CLASS_NAMES[predicted.item()]

    confidence = round(confidence.item() * 100, 2)

    return disease, confidence

#testing

if __name__ == "__main__":

    test_image = input("Enter Image Path : ")

    disease, confidence = predict_image(test_image)

    print("\nPrediction :", disease)
    print("Confidence :", confidence, "%")