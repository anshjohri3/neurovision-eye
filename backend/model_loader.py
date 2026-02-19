"""Model loading and management."""

import logging
import tensorflow as tf
from .config import MODEL_PATH

logger = logging.getLogger("neurodetect")


def load_model():
    logger.info(f"MODEL_PATH resolved to: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}. "
            "Make sure it's committed to GitHub under backend/models/."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

