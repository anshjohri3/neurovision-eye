"""Configuration constants for the brain tumor detector."""

from pathlib import Path
import os


# Model configuration
MODEL_PATH = Path(__file__).parent / "models" / "brain_tumor_multiclass.keras"
MODEL_PATH_LEGACY = Path(__file__).parent / "models" / "brain_tumor_ft.keras"

IMG_SIZE = (224, 224)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)


# API configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


# Gemini API configuration (FROM ENV)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Class display information
CLASS_INFO = {
    "glioma": {
        "display": "Glioma",
        "type": "tumor",
        "description": "Brain tumor - Glioma",
    },
    "meningioma": {
        "display": "Meningioma",
        "type": "tumor",
        "description": "Brain tumor - Meningioma",
    },
    "notumor": {
        "display": "No Tumor",
        "type": "normal",
        "description": "No tumor detected",
    },
    "pituitary": {
        "display": "Pituitary",
        "type": "tumor",
        "description": "Brain tumor - Pituitary",
    },
}
