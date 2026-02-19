"""FastAPI application for brain tumor detection."""

import logging
import os

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import CLASS_INFO, CLASS_NAMES, GEMINI_API_KEY as CONFIG_GEMINI_API_KEY, NUM_CLASSES
from .model_loader import load_model
from .utils import generate_heatmap_image, preprocess_image

# --------------------------------------------------
# App + logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurodetect")

app = FastAPI()

# allow frontend cross-origin requests (tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Gemini config
GEMINI_API_KEY = CONFIG_GEMINI_API_KEY  # already validated in config.py (env-based)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# Model loading (safe startup)-
model = None
MODEL_LOAD_ERROR = None


@app.on_event("startup")
def startup():
    global model, MODEL_LOAD_ERROR
    try:
        logger.info("Loading ML model...")
        model = load_model()
        logger.info("Model loaded successfully ")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        logger.exception("Model failed to load")
        # Keep app alive so /health shows the error



# Global exception handler (prevents mystery 502s)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    context: dict | None = None


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": MODEL_LOAD_ERROR,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor class from MRI image (4 classes: glioma, meningioma, notumor, pituitary).
    Returns probabilities for all classes.
    """
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        logger.info(f"/predict: filename={file.filename} bytes={len(contents)}")

        # Preprocess image
        x = preprocess_image(contents)
        logger.info(f"/predict: preprocess ok shape={getattr(x, 'shape', None)}")

        # Predict (4-class probabilities)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        # Build response with all class probabilities
        predictions = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_prob = float(probs[i])
            predictions[class_name] = {
                "probability": round(class_prob, 4),
                "confidence_pct": round(class_prob * 100, 2),
                "display": CLASS_INFO[class_name]["display"],
                "type": CLASS_INFO[class_name]["type"],
            }

        return {
            "predicted_class": pred_class,
            "predicted_display": CLASS_INFO[pred_class]["display"],
            "confidence_pct": round(confidence * 100, 2),
            "confidence": round(confidence, 4),
            "all_predictions": predictions,
            "is_tumor": pred_class != "notumor",
            "message": (
                f"Tumor detected: {CLASS_INFO[pred_class]['display']} "
                f"(confidence: {round(confidence * 100, 2)}%)"
                if pred_class != "notumor"
                else f"No tumor detected (confidence: {round(confidence * 100, 2)}%)"
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during /predict")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/heatmap")
@app.post("/api/heatmap")
async def get_heatmap(file: UploadFile = File(...)):
    """Generate a Grad-CAM heatmap showing model attention regions."""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_LOAD_ERROR}")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file upload")

        logger.info(f"/heatmap: filename={file.filename} bytes={len(contents)}")
        heatmap_b64 = generate_heatmap_image(model, contents)
        return {"heatmap": heatmap_b64}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during /heatmap")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
@app.post("/api/chat")
async def chat_with_assistant(request: ChatRequest):
    """Chat with AI assistant about brain tumor detection results."""

    if not GEMINI_API_KEY:
        return {
            "response": "AI explanation service is not configured.",
            "status": "degraded",
            "error": "missing_api_key"
        }

    system_prompt = (
        "You are a helpful medical AI assistant for NeuroDetect, a brain tumor detection system. "
        "You help users understand their MRI scan analysis results. "
        "IMPORTANT: This is for educational purposes only and not a medical diagnosis."
    )

    user_message = request.message
    if request.context:
        user_message = f"{user_message}\n\nCurrent scan results: {request.context}"

    prompt = f"{system_prompt}\n\nUser: {user_message}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,  # lower to reduce quota burn
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload, timeout=30.0)

        if resp.status_code != 200:
            logger.warning(f"Gemini unavailable ({resp.status_code})")
            return {
                "response": (
                    "AI explanation is temporarily unavailable. "
                    "Your scan prediction was completed successfully."
                ),
                "status": "degraded",
                "error": "ai_unavailable"
            }

        data = resp.json()

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not text:
            return {
                "response": "AI explanation is currently unavailable.",
                "status": "degraded",
                "error": "empty_ai_response"
            }

        reply = text.strip()
        return {
            "response": reply,
            "status": "ok",
            "error": None
        }

    except Exception as e:
        logger.exception("Unexpected error during /chat")
        return {
            "response": (
                "AI explanation service is currently unavailable. "
                "Please try again later."
            ),
            "status": "error",
            "error": "internal_error"
        }
