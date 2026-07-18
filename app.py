#all library

from pathlib import Path
import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from prediction.predict import predict_image
from utils.disease_info import DISEASE_INFO


#  app creation
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)




@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No file selected."

    filename = secure_filename(file.filename)

    image_path = UPLOAD_FOLDER / filename

    file.save(image_path)

    disease, confidence = predict_image(str(image_path))

    info = DISEASE_INFO.get(
        disease,
        {
            "description": "No description available.",
            "treatment": "No treatment available.",
            "prevention": "No prevention available.",
            "fertilizer": "No recommendation available."
        }
    )

    return render_template(
        "result.html",
        image_path=f"/static/uploads/{filename}",
        disease=disease,
        confidence=confidence,
        description=info["description"],
        treatment=info["treatment"],
        prevention=info["prevention"],
        fertilizer=info["fertilizer"]
    )




if __name__ == "__main__":
    app.run(debug=True)