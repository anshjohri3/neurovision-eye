"""Utility functions for image processing and model interpretation."""

import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
from .config import IMG_SIZE


def preprocess_image(img_bytes):
    """
    Load and preprocess image bytes to model input format.
    
    Args:
        img_bytes: Raw image bytes
        
    Returns:
        Preprocessed image array in shape (1, 224, 224, 3) with [-1, 1] range
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    
    # Normalize to [0, 1]
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # (1, 224, 224, 3)
    
    # MobileNetV2 expects [-1, 1]; convert from [0, 1]
    x = (x * 2.0) - 1.0
    
    return x


def generate_heatmap_image(model, img_bytes):
    """
    Generate Grad-CAM heatmap using input-level gradients.
    Shows regions most important for model's classification decision.
    Preserves original image dimensions.
    
    Args:
        model: Keras model for tumor detection
        img_bytes: Raw image bytes
        
    Returns:
        Base64-encoded PNG string with heatmap overlay
    """
    try:
        # Load original image at full resolution
        img_orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_size = img_orig.size  # (width, height)
        img_array_orig = np.array(img_orig, dtype=np.uint8)
        
        # Preprocess for model (224x224)
        x = preprocess_image(img_bytes)
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Get prediction and compute gradients
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            logits = model(x_tensor, training=False)
        
        predictions = logits.numpy()[0]
        pred_idx = int(np.argmax(predictions))
        
        # Get gradients of predicted class w.r.t. model input (224x224)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            logits = model(x_tensor, training=False)
            class_score = logits[0, pred_idx]
        
        grads = tape.gradient(class_score, x_tensor)
        
        # Prepare base image (use original size, grayscale)
        img_gray_orig = np.mean(img_array_orig.astype(np.float32), axis=2)
        result_img = np.stack([img_gray_orig, img_gray_orig, img_gray_orig], axis=2)
        
        if grads is not None:
            # Convert gradients to saliency map (224x224)
            grads_np = grads.numpy()[0]  # (224, 224, 3)
            saliency = np.mean(np.abs(grads_np), axis=2)  # (224, 224)
            saliency = saliency / (np.max(saliency) + 1e-8)
            
            # Resize saliency map to original image size
            from scipy.ndimage import zoom
            orig_h, orig_w = orig_size[1], orig_size[0]
            scale_h = orig_h / 224.0
            scale_w = orig_w / 224.0
            saliency_resized = zoom(saliency, (scale_h, scale_w), order=1)
            
            # Create threshold for HIGH attention (top 10% most important)
            threshold = np.percentile(saliency_resized, 90)
            attention_mask = (saliency_resized > threshold).astype(np.float32)
            
            # Apply red overlay ONLY where attention is high
            for i in range(orig_h):
                for j in range(orig_w):
                    if attention_mask[i, j] > 0:
                        # Red intensity increases with attention
                        intensity = saliency_resized[i, j]
                        result_img[i, j] = [255, 50 * (1 - intensity), 50 * (1 - intensity)]
        
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)
        
        # Encode to base64
        vis_img_pil = Image.fromarray(result_img)
        buffer = io.BytesIO()
        vis_img_pil.save(buffer, format="PNG")
        buffer.seek(0)
        b64_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64_str}"
        
    except Exception as e:
        print(f"Error generating heatmap visualization: {e}")
        import traceback
        traceback.print_exc()
        raise
