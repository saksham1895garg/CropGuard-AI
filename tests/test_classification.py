"""
Unit tests for CropGuard AI foliar pathology classification subsystem.
Verifies class mapping, group-aware split loading, class-weight calculus,
model-specific preprocessing, neural network output dimensions, and evaluation metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.classification.models import (
    build_custom_foliar_cnn,
    build_mobilenet_v2_classifier,
    build_resnet50_classifier,
    set_backbone_trainable_layers,
)
from src.cropguard.classification.preprocessing import (
    compute_class_weights,
    get_model_preprocessing,
    get_plantvillage_classes,
)
from src.cropguard.classification.evaluation import compute_classification_metrics
from src.cropguard.data.loader import (
    load_dataset_records,
    build_classification_dataset,
    get_stratified_subset,
)


def test_class_mapping_consistency():
    """Verify that PlantVillage class mapping contains exactly 38 sorted classes with bidirectional integrity."""
    classes, class_to_idx, idx_to_class = get_plantvillage_classes()
    assert len(classes) == 38, f"Expected 38 classes, found {len(classes)}"
    assert classes == sorted(classes), "Classes must be alphabetically sorted"
    assert len(class_to_idx) == 38
    assert len(idx_to_class) == 38

    # Verify bidirectional identity
    for idx, name in enumerate(classes):
        assert class_to_idx[name] == idx
        assert idx_to_class[idx] == name


def test_group_aware_csv_loading():
    """Verify loading from group-aware train.csv split file."""
    train_csv = PROJECT_ROOT / "data" / "splits" / "classification_group_aware" / "train.csv"
    assert train_csv.exists(), f"Split file missing: {train_csv}"

    filepaths, labels, class_to_idx = load_dataset_records(train_csv)
    assert len(filepaths) == 35221, f"Expected 35,221 train samples, got {len(filepaths)}"
    assert len(labels) == 35221
    assert len(class_to_idx) == 38
    assert min(labels) == 0
    assert max(labels) == 37


def test_class_weight_calculation():
    """Verify inverse-frequency class weights satisfy sum(w_k * N_k) == N."""
    train_csv = PROJECT_ROOT / "data" / "splits" / "classification_group_aware" / "train.csv"
    _, class_to_idx, _ = get_plantvillage_classes()
    weights = compute_class_weights(train_csv, class_to_idx)

    assert len(weights) == 38
    for cls_idx, w in weights.items():
        assert w > 0.0, f"Class weight for {cls_idx} must be positive, got {w}"

    # Load counts to verify exact mathematical balance: sum(w_k * N_k) == N
    _, labels, _ = load_dataset_records(train_csv)
    from collections import Counter
    counts = Counter(labels)
    N = len(labels)
    weighted_sum = sum(weights[k] * counts[k] for k in range(38))
    np.testing.assert_allclose(weighted_sum, N, rtol=1e-5)


def test_model_preprocessing_functions():
    """Verify distinct model-specific preprocessing transforms."""
    # Custom CNN: [0, 255] -> [0.0, 1.0]
    custom_fn = get_model_preprocessing("custom_cnn")
    sample_tensor = tf.constant([[[0.0, 127.5, 255.0]]], dtype=tf.float32)
    custom_out = custom_fn(sample_tensor).numpy()
    np.testing.assert_allclose(custom_out, [[[0.0, 0.5, 1.0]]], atol=1e-5)

    # MobileNetV2: [0, 255] -> [-1.0, 1.0]
    mb_fn = get_model_preprocessing("mobilenet_v2")
    mb_out = mb_fn(sample_tensor).numpy()
    np.testing.assert_allclose(mb_out, [[[-1.0, 0.0, 1.0]]], atol=1e-4)

    # ResNet50: RGB -> BGR with channel mean subtraction
    rn_fn = get_model_preprocessing("resnet50")
    rn_out = rn_fn(sample_tensor).numpy()
    # Input is RGB [0, 127.5, 255.0]. BGR is [255.0, 127.5, 0.0].
    # Subtraction: B - 103.939 = 151.061, G - 116.779 = 10.721, R - 123.68 = -123.68
    expected_bgr = np.array([[[151.061, 10.721, -123.68]]], dtype=np.float32)
    np.testing.assert_allclose(rn_out, expected_bgr, atol=1e-2)


def test_custom_foliar_cnn_dimensions():
    """Verify custom CNN layer outputs and 38-class softmax prediction head."""
    model = build_custom_foliar_cnn(input_shape=(224, 224, 3), num_classes=38)
    assert model.output_shape == (None, 38)
    assert model.input_shape == (None, 224, 224, 3)

    # Forward pass test with dummy batch
    dummy = tf.zeros((2, 224, 224, 3), dtype=tf.float32)
    preds = model(dummy, training=False).numpy()
    assert preds.shape == (2, 38)
    # Verify valid probability distribution (sum to 1)
    np.testing.assert_allclose(np.sum(preds, axis=1), [1.0, 1.0], atol=1e-5)


def test_mobilenet_v2_classifier_dimensions():
    """Verify MobileNetV2 feature extractor and 38-class classification head."""
    model, backbone = build_mobilenet_v2_classifier(
        input_shape=(224, 224, 3),
        num_classes=38,
        pretrained=False,
        freeze_backbone=True
    )
    assert model.output_shape == (None, 38)
    assert not backbone.trainable, "Backbone must be frozen initially"

    # Test fine-tune unfreezing helper
    trainable_weights = set_backbone_trainable_layers(backbone, num_layers_to_unfreeze=10)
    assert trainable_weights > 0, "Top 10 layers should have trainable parameters"


def test_resnet50_classifier_dimensions():
    """Verify ResNet50 classifier output shape."""
    model, backbone = build_resnet50_classifier(
        input_shape=(224, 224, 3),
        num_classes=38,
        pretrained=False,
        freeze_backbone=True
    )
    assert model.output_shape == (None, 38)
    assert not backbone.trainable


def test_classification_metrics_math():
    """Verify multi-class evaluation metric calculations on synthetic data."""
    # Synthetic: 4 classes, 8 samples
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    # Predictions: 6 correct, 2 mistakes
    y_pred = np.array([0, 1, 2, 0, 0, 1, 2, 2])

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=4)
    assert metrics["accuracy"] == 6.0 / 8.0
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["confusion_matrix"].shape == (4, 4)
    assert metrics["total_samples"] == 8


def test_stratified_subset_all_classes_represented():
    """Verify that get_stratified_subset guarantees representation of all classes present."""
    train_csv = PROJECT_ROOT / "data" / "splits" / "classification_group_aware" / "train.csv"
    assert train_csv.exists(), f"Split file missing: {train_csv}"

    filepaths, labels, class_to_idx = load_dataset_records(train_csv)
    assert len(filepaths) == 35221
    assert len(class_to_idx) == 38

    # Request small subset (e.g. 512 samples)
    sub_fps, sub_lbls = get_stratified_subset(filepaths, labels, target_samples=512, seed=42)

    # Check that lengths match
    assert len(sub_fps) == len(sub_lbls)
    # Check that all 38 classes are represented (no class starvation)
    unique_classes_in_subset = set(sub_lbls)
    assert len(unique_classes_in_subset) == 38, (
        f"Expected all 38 classes to be represented in stratified subset, got {len(unique_classes_in_subset)}"
    )

    # Check subset size is close to target (within small rounding margin for 38 classes)
    assert 500 <= len(sub_fps) <= 530

    # Check determinism with identical seed
    sub_fps2, sub_lbls2 = get_stratified_subset(filepaths, labels, target_samples=512, seed=42)
    assert sub_fps == sub_fps2
    assert sub_lbls == sub_lbls2

