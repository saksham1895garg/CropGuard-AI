"""
CropGuard AI - Production Inference Engine for Classification & Lesion Segmentation.
Combines ResNet50 (38-class leaf pathology) and U-Net (foliar lesion segmentation)
with standardized EXIF handling, preprocessing, and overlay generation.
"""

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_RESNET_PATH = PROJECT_ROOT / "models" / "classification" / "production" / "resnet50" / "best_model.keras"
DEFAULT_UNET_PATH = PROJECT_ROOT / "models" / "segmentation" / "production" / "unet" / "best_model.keras"
DEFAULT_YOLO_PATH = PROJECT_ROOT / "yolo26n.pt"


def configure_gpu_memory():
    """Configures dynamic GPU memory growth to prevent VRAM allocation conflicts."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


configure_gpu_memory()

PLANTVILLAGE_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_and_orient_image(
    image_input: Union[bytes, str, Path, io.BytesIO, Image.Image]
) -> Image.Image:
    """
    Loads an image from bytes, filepath, or file-like buffer,
    normalizes camera EXIF orientation flags, and converts to RGB format.

    Args:
        image_input: Raw image bytes, filepath, file buffer, or PIL Image.

    Returns:
        RGB PIL Image with normalized orientation.
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input))
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    else:
        pil_img = Image.open(image_input)

    # Normalize EXIF orientation (e.g. tag 274: 6 on mobile cameras)
    transposed_img = ImageOps.exif_transpose(pil_img)
    if transposed_img is not None:
        pil_img = transposed_img

    # Convert to standard 3-channel RGB
    return pil_img.convert("RGB")


def preprocess_classification_input(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Applies exact ResNet50 preprocessing:
    1. Resizes image to target spatial dimensions (224, 224) via bilinear interpolation.
    2. Converts to float32 NumPy array.
    3. Converts RGB to BGR and subtracts ImageNet channel means:
       [103.939, 116.779, 123.68] (via tf.keras.applications.resnet50.preprocess_input).
    4. Expands batch dimension -> (1, 224, 224, 3).
    """
    resized = image.resize(target_size, Image.Resampling.BILINEAR)
    img_arr = np.array(resized, dtype=np.float32)
    # tf.keras resnet50 preprocess_input expects RGB in [0, 255] and converts to BGR with mean subtraction
    preprocessed = tf.keras.applications.resnet50.preprocess_input(img_arr)
    return np.expand_dims(preprocessed, axis=0)


def preprocess_segmentation_input(
    image: Image.Image,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Applies exact U-Net segmentation preprocessing:
    1. Resizes image to target spatial dimensions (256, 256) via bilinear interpolation.
    2. Converts to float32 NumPy array.
    3. Scales pixel values linearly to [0.0, 1.0].
    4. Expands batch dimension -> (1, 256, 256, 3).
    """
    resized = image.resize(target_size, Image.Resampling.BILINEAR)
    img_arr = np.array(resized, dtype=np.float32) / 255.0
    img_arr = np.clip(img_arr, 0.0, 1.0)
    return np.expand_dims(img_arr, axis=0)


def load_classification_model(
    model_path: Optional[Union[str, Path]] = None
) -> tf.keras.Model:
    """Loads the production ResNet50 classification model."""
    path = Path(model_path) if model_path else DEFAULT_RESNET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Classification model checkpoint not found at: {path}")
    return keras.models.load_model(str(path))


def load_segmentation_model(
    model_path: Optional[Union[str, Path]] = None
) -> tf.keras.Model:
    """Loads the production U-Net segmentation model with required custom objects."""
    path = Path(model_path) if model_path else DEFAULT_UNET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Segmentation model checkpoint not found at: {path}")

    from src.cropguard.segmentation.training import (
        BCEDiceLoss,
        bce_dice_loss,
        dice_loss,
    )
    from src.cropguard.segmentation.evaluation import (
        dice_coefficient,
        iou_score,
    )

    custom_objects = {
        "BCEDiceLoss": BCEDiceLoss,
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
    }
    return keras.models.load_model(str(path), custom_objects=custom_objects)


def format_class_name(raw_name: str) -> Dict[str, Any]:
    """
    Parses a raw PlantVillage class name into human-readable crop and condition labels.
    Example:
        'Tomato___Late_blight' -> {
            'crop': 'Tomato',
            'condition': 'Late Blight',
            'full': 'Tomato — Late Blight',
            'is_healthy': False
        }
    """
    if "___" in raw_name:
        crop_part, condition_part = raw_name.split("___", 1)
    else:
        crop_part = raw_name
        condition_part = ""

    crop_clean = crop_part.replace("_", " ").strip()
    # Format common plant name conventions
    if crop_clean.lower().startswith("cherry (including sour)"):
        crop_clean = "Cherry"
    elif crop_clean.lower().startswith("pepper, bell"):
        crop_clean = "Bell Pepper"
    elif crop_clean.lower().startswith("corn (maize)"):
        crop_clean = "Corn (Maize)"

    condition_clean = condition_part.replace("_", " ").strip()
    is_healthy = condition_clean.lower() == "healthy"

    if is_healthy:
        formatted_condition = "Healthy Leaf"
    else:
        formatted_condition = condition_clean.title()

    full_display = f"{crop_clean} — {formatted_condition}"

    return {
        "raw": raw_name,
        "crop": crop_clean,
        "condition": formatted_condition,
        "full_display": full_display,
        "is_healthy": is_healthy,
    }


def predict_classification(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Performs ResNet50 inference and extracts top-k predictions with confidence scores.
    """
    if class_names is None:
        class_names = PLANTVILLAGE_CLASSES

    t0 = time.perf_counter()
    if callable(model):
        raw_tensor = model(preprocessed_image, training=False)
        raw_preds = raw_tensor.numpy() if hasattr(raw_tensor, "numpy") else np.array(raw_tensor)
    else:
        raw_preds = model.predict(preprocessed_image, verbose=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = raw_preds[0]
    # Ensure probabilities sum to 1.0
    if not np.allclose(np.sum(probs), 1.0, atol=1e-3):
        # Softmax if raw logits were output
        probs = np.exp(probs - np.max(probs))
        probs = probs / np.sum(probs)

    top_indices = np.argsort(probs)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        raw_class = class_names[idx]
        conf = float(probs[idx])
        parsed = format_class_name(raw_class)
        results.append({
            "rank": rank + 1,
            "raw_class": raw_class,
            "crop": parsed["crop"],
            "condition": parsed["condition"],
            "full_display": parsed["full_display"],
            "is_healthy": parsed["is_healthy"],
            "confidence": conf,
            "confidence_pct": conf * 100.0,
        })

    return {
        "top_prediction": results[0],
        "top_k": results,
        "latency_ms": latency_ms,
    }


def predict_segmentation(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Performs U-Net segmentation inference and generates binary lesion mask and severity metrics.
    """
    t0 = time.perf_counter()
    if callable(model):
        raw_tensor = model(preprocessed_image, training=False)
        pred_prob_map = raw_tensor.numpy() if hasattr(raw_tensor, "numpy") else np.array(raw_tensor)
    else:
        pred_prob_map = model.predict(preprocessed_image, verbose=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Squeeze to (H, W)
    prob_map = np.squeeze(pred_prob_map).astype(np.float32)
    binary_mask = (prob_map >= threshold).astype(np.uint8)

    total_pixels = binary_mask.size
    lesion_pixels = int(np.sum(binary_mask))
    lesion_coverage_pct = float((lesion_pixels / total_pixels) * 100.0)

    return {
        "prob_map": prob_map,
        "binary_mask": binary_mask,
        "threshold": threshold,
        "total_pixels": total_pixels,
        "lesion_pixels": lesion_pixels,
        "lesion_coverage_pct": lesion_coverage_pct,
        "latency_ms": latency_ms,
    }


def create_segmentation_overlay(
    original_image: Image.Image,
    binary_mask: np.ndarray,
    color: Tuple[int, int, int] = (220, 38, 38),  # Crimson red
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blends a semi-transparent colored mask over the original leaf image.

    Args:
        original_image: Original RGB PIL image.
        binary_mask: 2D NumPy array {0, 1} representing predicted lesions.
        color: RGB tuple for lesion overlay.
        alpha: Transparency value [0.0, 1.0].

    Returns:
        Blended RGB PIL Image.
    """
    orig_w, orig_h = original_image.size

    # Convert binary mask to PIL and resize to original image dimensions using NEAREST
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")
    mask_resized = mask_pil.resize((orig_w, orig_h), Image.Resampling.NEAREST)
    mask_arr = np.array(mask_resized) > 127

    # Build RGBA colored overlay
    overlay_arr = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    overlay_arr[mask_arr, 0] = color[0]
    overlay_arr[mask_arr, 1] = color[1]
    overlay_arr[mask_arr, 2] = color[2]
    overlay_arr[mask_arr, 3] = int(255 * alpha)

    overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
    base_rgba = original_image.convert("RGBA")
    blended_rgba = Image.alpha_composite(base_rgba, overlay_img)

    return blended_rgba.convert("RGB")


# ==============================================================================
# YOLO Pest Detection Exports
# ==============================================================================

def load_pest_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    """Loads YOLO pest detection model (fine-tuned checkpoint or base YOLO26n)."""
    from src.cropguard.detection.models import load_pest_detector
    return load_pest_detector(model_path)


def predict_pest_detection(
    model: Any,
    image_input: Union[Image.Image, np.ndarray, str, Path],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """Runs YOLO pest detection inference."""
    from src.cropguard.detection.inference import predict_pests
    return predict_pests(
        model=model,
        image_input=image_input,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


def render_pest_overlay(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    box_color: Tuple[int, int, int] = (220, 38, 38),
    line_width: int = 3,
) -> Image.Image:
    """Renders pest bounding box overlay on leaf image."""
    from src.cropguard.detection.inference import render_pest_detections
    return render_pest_detections(
        image=image,
        detections=detections,
        box_color=box_color,
        line_width=line_width,
    )

