"""
Unit tests for CropGuard AI inference engine and Streamlit app utilities.
Tests image loading, EXIF orientation correction, preprocessing pipelines,
prediction formatting, lesion area calculations, and overlay blending.
"""

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.inference import (
    load_and_orient_image,
    preprocess_classification_input,
    preprocess_segmentation_input,
    format_class_name,
    predict_classification,
    predict_segmentation,
    create_segmentation_overlay,
    load_classification_model,
    load_segmentation_model,
    PLANTVILLAGE_CLASSES,
)


@pytest.fixture
def sample_rgb_image():
    """Generates a synthetic RGB leaf image for testing."""
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    arr[:, :, 1] = 180  # Green dominant leaf body
    arr[100:150, 150:200] = [139, 69, 19]  # Brown lesion patch
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def synthetic_exif_image():
    """Generates a synthetic image with EXIF orientation tag 274 = 6 (90 CW rotation)."""
    img = Image.new("RGB", (200, 100), color=(50, 150, 50))
    exif = img.getexif()
    exif[274] = 6  # EXIF orientation tag 6: Rotated 90 CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    return buf


def test_load_and_orient_image_from_pil(sample_rgb_image):
    """Verifies load_and_orient_image handles raw PIL objects correctly."""
    oriented = load_and_orient_image(sample_rgb_image)
    assert isinstance(oriented, Image.Image)
    assert oriented.mode == "RGB"
    assert oriented.size == (400, 300)


def test_load_and_orient_image_from_bytes(sample_rgb_image):
    """Verifies load_and_orient_image handles in-memory byte buffers."""
    buf = io.BytesIO()
    sample_rgb_image.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    oriented = load_and_orient_image(raw_bytes)
    assert isinstance(oriented, Image.Image)
    assert oriented.mode == "RGB"
    assert oriented.size == (400, 300)


def test_load_and_orient_image_exif_normalization(synthetic_exif_image):
    """Verifies EXIF tag 274: 6 is transposed so width and height are correctly swapped."""
    # Input was saved as (200, 100) with orientation tag 6 (rotate 90 CW)
    # Transposed image should become (100, 200)
    oriented = load_and_orient_image(synthetic_exif_image)
    assert oriented.size == (100, 200)
    assert oriented.mode == "RGB"


def test_preprocess_classification_input(sample_rgb_image):
    """Verifies ResNet50 preprocessing produces correct shape and Caffe normalization."""
    tensor = preprocess_classification_input(sample_rgb_image, target_size=(224, 224))
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    # ResNet50 caffe preprocessing subtracts ImageNet means, resulting in negative values
    assert tensor.min() < 0.0


def test_preprocess_segmentation_input(sample_rgb_image):
    """Verifies U-Net preprocessing produces (1, 256, 256, 3) normalized in [0.0, 1.0]."""
    tensor = preprocess_segmentation_input(sample_rgb_image, target_size=(256, 256))
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 256, 256, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_format_class_name():
    """Verifies class name string parsing into crop and disease condition."""
    res1 = format_class_name("Tomato___Late_blight")
    assert res1["crop"] == "Tomato"
    assert res1["condition"] == "Late Blight"
    assert res1["full_display"] == "Tomato — Late Blight"
    assert res1["is_healthy"] is False

    res2 = format_class_name("Apple___healthy")
    assert res2["crop"] == "Apple"
    assert res2["condition"] == "Healthy Leaf"
    assert res2["is_healthy"] is True

    res3 = format_class_name("Pepper,_bell___Bacterial_spot")
    assert res3["crop"] == "Bell Pepper"
    assert res3["condition"] == "Bacterial Spot"
    assert res3["is_healthy"] is False

    res4 = format_class_name("Cherry_(including_sour)___Powdery_mildew")
    assert res4["crop"] == "Cherry"
    assert res4["condition"] == "Powdery Mildew"
    assert res4["is_healthy"] is False


def test_predict_classification_synthetic():
    """Verifies classification output structure and top-k sorting on a synthetic model."""
    num_classes = len(PLANTVILLAGE_CLASSES)
    assert num_classes == 38

    # Create dummy Keras model outputting uniform or peaked logits
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    dummy_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32)
    res = predict_classification(dummy_model, dummy_input, top_k=3)

    assert "top_prediction" in res
    assert "top_k" in res
    assert "latency_ms" in res
    assert len(res["top_k"]) == 3
    assert res["top_k"][0]["confidence"] >= res["top_k"][1]["confidence"]
    assert res["top_k"][1]["confidence"] >= res["top_k"][2]["confidence"]


def test_predict_segmentation_synthetic():
    """Verifies segmentation probability binarization and severity calculation."""
    # Create dummy model returning 256x256x1 probabilities
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(inputs)
    dummy_model = tf.keras.Model(inputs=inputs, outputs=x)

    # Input with 0.5 everywhere except one quarter with 1.0
    dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    res = predict_segmentation(dummy_model, dummy_input, threshold=0.5)

    assert "prob_map" in res
    assert "binary_mask" in res
    assert "lesion_coverage_pct" in res
    assert res["prob_map"].shape == (256, 256)
    assert res["binary_mask"].shape == (256, 256)
    assert set(np.unique(res["binary_mask"])).issubset({0, 1})
    assert 0.0 <= res["lesion_coverage_pct"] <= 100.0


def test_create_segmentation_overlay(sample_rgb_image):
    """Verifies overlay generation preserves input dimensions and creates valid RGB image."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:128, 64:128] = 1  # 25% of mask in central block

    overlay = create_segmentation_overlay(
        original_image=sample_rgb_image,
        binary_mask=mask,
        color=(220, 38, 38),
        alpha=0.5,
    )

    assert isinstance(overlay, Image.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == sample_rgb_image.size  # Matches original image (400, 300)


def test_production_model_checkpoints_exist():
    """Verifies that the frozen production model checkpoints are present and readable."""
    from src.cropguard.inference import DEFAULT_RESNET_PATH, DEFAULT_UNET_PATH, DEFAULT_YOLO_PATH
    assert DEFAULT_RESNET_PATH.exists(), f"ResNet50 missing at {DEFAULT_RESNET_PATH}"
    assert DEFAULT_UNET_PATH.exists(), f"U-Net missing at {DEFAULT_UNET_PATH}"
    assert DEFAULT_YOLO_PATH.exists(), f"YOLO base missing at {DEFAULT_YOLO_PATH}"


def test_pest_detection_inference_pipeline(sample_rgb_image):
    """Verifies end-to-end pest detection inference export."""
    from src.cropguard.inference import load_pest_model, predict_pest_detection, render_pest_overlay
    model = load_pest_model()
    res = predict_pest_detection(model, sample_rgb_image, conf_threshold=0.25)

    assert "num_detections" in res
    assert "detections" in res
    assert "status_message" in res
    assert "latency_ms" in res
    assert isinstance(res["detections"], list)

    overlay = render_pest_overlay(sample_rgb_image, res["detections"])
    assert isinstance(overlay, Image.Image)
    assert overlay.size == sample_rgb_image.size

