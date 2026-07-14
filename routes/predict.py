import os

from flask import (
    Blueprint,
    request,
    render_template,
    current_app
)

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():

    image = request.files.get("image")

    if image is None or image.filename == "":
        return "No image selected."

    upload_folder = current_app.config["UPLOAD_FOLDER"]

    os.makedirs(upload_folder, exist_ok=True)

    image_path = os.path.join(upload_folder, image.filename)

    image.save(image_path)

    return render_template(
        "result.html",
        image_name=image.filename,
        disease="Prediction Coming Soon",
        confidence="--"
    )