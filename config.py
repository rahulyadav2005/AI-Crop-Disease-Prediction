import os

class Config:

    SECRET_KEY = "crop_disease_secret_key"

    UPLOAD_FOLDER = "static/uploads/original"

    PROCESSED_FOLDER = "static/uploads/processed"

    MODEL_PATH = "model/crop_model.keras"

    DATABASE = "database/database.db"

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}