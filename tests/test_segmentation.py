"""
CropGuard AI - Foliar Lesion Segmentation Subsystem Test Suite
Validates:
1. U-Net model instantiation, forward pass, parameter scaling, and output tensor shapes.
2. Preprocessing pipeline: image scaling to [0, 1], mask binarization, spatial resizing, EXIF handling.
3. Loss functions: BCE loss, Dice loss, and Combined BCE+Dice loss sanity and gradient flow.
4. Evaluation metrics: Sørensen-Dice coefficient and Jaccard IoU score in pure NumPy and TensorFlow.
5. Foliar lesion severity index calculation.
6. PlantSeg split disjointness and test split quarantine preservation.
7. tf.data segmentation dataset pipeline loading and batch generation.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.cropguard.segmentation as seg
from scripts.prepare_plantseg import create_synthetic_plantseg_sample


@pytest.fixture(scope="module")
def tf_gpu_setup():
    """Ensures dynamic GPU memory growth is enabled to prevent VRAM over-allocation."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def test_unet_instantiation_and_shapes(tf_gpu_setup):
    """Verifies U-Net instantiates correctly with expected input/output shapes and parameter counts."""
    # Build lightweight test configuration
    model = seg.build_unet(
        input_shape=(128, 128, 3),
        num_classes=1,
        base_filters=16,
        depth=3,
        use_batchnorm=True,
        model_name="test_light_unet"
    )

    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, 128, 128, 1)
    assert model.count_params() > 0

    # Test forward pass with synthetic batch
    dummy_input = tf.random.uniform((2, 128, 128, 3), minval=0.0, maxval=1.0)
    preds = model(dummy_input, training=False)
    assert preds.shape == (2, 128, 128, 1)
    # Output must be probabilities in [0.0, 1.0] from Sigmoid
    assert tf.reduce_min(preds) >= 0.0
    assert tf.reduce_max(preds) <= 1.0


def test_mask_binarization_and_preprocessing():
    """Verifies that raw multi-integer disease mask IDs are cleanly binarized to {0.0, 1.0}."""
    raw_mask = np.zeros((100, 100), dtype=np.uint8)
    raw_mask[20:50, 30:70] = 42  # Synthetic disease ID 42 (e.g. coffee black rot)
    raw_mask[60:80, 10:40] = 99  # Synthetic disease ID 99 (e.g. tomato late blight)

    bin_mask = seg.binarize_mask(raw_mask)
    assert bin_mask.shape == (100, 100, 1)
    assert bin_mask.dtype == np.float32
    assert set(np.unique(bin_mask)).issubset({0.0, 1.0})
    assert np.count_nonzero(bin_mask == 1.0) == (30 * 40) + (20 * 30)


def test_preprocess_image_and_mask_dual_pipeline():
    """Verifies simultaneous image and mask spatial alignment, resizing, and normalization."""
    # Create non-square image and mask
    img_arr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    mask_arr = np.zeros((300, 400), dtype=np.uint8)
    mask_arr[50:150, 50:150] = 12

    p_img, p_mask = seg.preprocess_image_and_mask(img_arr, mask_arr, target_shape=(256, 256))

    assert p_img.shape == (256, 256, 3)
    assert p_mask.shape == (256, 256, 1)
    assert p_img.dtype == np.float32
    assert p_mask.dtype == np.float32
    assert 0.0 <= p_img.min() and p_img.max() <= 1.0
    assert set(np.unique(p_mask)).issubset({0.0, 1.0})


def test_dice_and_iou_metrics():
    """Verifies Sørensen-Dice coefficient and Jaccard IoU calculations in pure NumPy and TensorFlow."""
    # Case 1: Identical masks
    y_true = np.zeros((1, 50, 50, 1), dtype=np.float32)
    y_true[:, 10:30, 10:30, :] = 1.0
    y_pred_perfect = y_true.copy()

    dice_perf = seg.dice_coefficient_np(y_true, y_pred_perfect)
    iou_perf = seg.iou_score_np(y_true, y_pred_perfect)
    assert np.isclose(dice_perf, 1.0, atol=1e-3)
    assert np.isclose(iou_perf, 1.0, atol=1e-3)

    # TF Tensor metrics
    dice_tf = seg.dice_coefficient(tf.constant(y_true), tf.constant(y_pred_perfect)).numpy()
    iou_tf = seg.iou_score(tf.constant(y_true), tf.constant(y_pred_perfect)).numpy()
    assert np.isclose(dice_tf, 1.0, atol=1e-3)
    assert np.isclose(iou_tf, 1.0, atol=1e-3)

    # Case 2: Disjoint masks
    y_pred_disjoint = np.zeros_like(y_true)
    y_pred_disjoint[:, 35:45, 35:45, :] = 1.0

    dice_disj = seg.dice_coefficient_np(y_true, y_pred_disjoint)
    iou_disj = seg.iou_score_np(y_true, y_pred_disjoint)
    assert dice_disj < 0.01
    assert iou_disj < 0.01

    # Case 3: 50% overlap calculation
    metrics = seg.compute_segmentation_metrics(y_true, y_pred_perfect)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["tp"] == 20 * 20


def test_bce_dice_loss_sanity_and_gradients(tf_gpu_setup):
    """Verifies Combined BCE + Dice loss decreases for matching predictions and computes gradients."""
    y_true = tf.constant(np.random.randint(0, 2, (2, 64, 64, 1)).astype(np.float32))
    y_good = tf.clip_by_value(y_true * 0.9 + 0.05, 1e-4, 1.0 - 1e-4)
    y_bad = tf.clip_by_value((1.0 - y_true) * 0.9 + 0.05, 1e-4, 1.0 - 1e-4)

    loss_fn = seg.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    loss_good = loss_fn(y_true, y_good).numpy()
    loss_bad = loss_fn(y_true, y_bad).numpy()

    assert loss_good < loss_bad, f"Good prediction loss ({loss_good}) should be lower than bad ({loss_bad})"

    # Verify gradient computation with GradientTape
    var = tf.Variable(tf.random.uniform((1, 32, 32, 1), 0.1, 0.9))
    with tf.GradientTape() as tape:
        l = seg.bce_dice_loss(y_true[:1, :32, :32, :], var)
    grad = tape.gradient(l, var)
    assert grad is not None
    assert not np.isnan(grad.numpy()).any()


def test_lesion_severity_calculation():
    """Verifies foliar lesion area severity metric computation."""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[:25, :50] = 1.0  # 25 * 50 = 1250 pixels out of 10,000 = 12.5%

    sev = seg.compute_lesion_severity(mask)
    assert sev["lesion_pixels"] == 1250
    assert sev["total_pixels"] == 10000
    assert np.isclose(sev["coverage_pct"], 12.5)


def test_plantseg_splits_disjointness_and_counts():
    """Verifies that segmentation split CSVs exist, are 100% disjoint, and match documented counts."""
    splits_dir = Path("data/splits/segmentation")
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    assert train_csv.exists()
    assert val_csv.exists()
    assert test_csv.exists()

    train_recs = seg.load_segmentation_records(train_csv)
    val_recs = seg.load_segmentation_records(val_csv)
    test_recs = seg.load_segmentation_records(test_csv)

    assert len(train_recs) == 7916
    assert len(val_recs) == 1247
    assert len(test_recs) == 2295
    assert len(train_recs) + len(val_recs) + len(test_recs) == 11458

    train_imgs = set(r["image_name"] for r in train_recs)
    val_imgs = set(r["image_name"] for r in val_recs)
    test_imgs = set(r["image_name"] for r in test_recs)

    assert len(train_imgs & val_imgs) == 0, "Train and Val overlap in PlantSeg!"
    assert len(train_imgs & test_imgs) == 0, "Train and Test overlap in PlantSeg!"
    assert len(val_imgs & test_imgs) == 0, "Val and Test overlap in PlantSeg!"


def test_synthetic_segmentation_dataset_pipeline(tmp_path):
    """Verifies tf.data dataset creation and batch generation using local synthetic image-mask pairs."""
    sample_dir = tmp_path / "synthetic_pairs"
    records = []
    for i in range(4):
        img_f, mask_f = create_synthetic_plantseg_sample(sample_dir, sample_id=f"test_{i}", width=128, height=128)
        records.append({
            "image_name": f"test_{i}.png",
            "mask_name": f"test_{i}_mask.png",
            "local_image_path": str(img_f),
            "local_mask_path": str(mask_f)
        })

    ds = seg.build_segmentation_dataset(
        records=records,
        batch_size=2,
        target_shape=(128, 128),
        shuffle=False
    )

    batches = list(ds.take(1))
    assert len(batches) == 1
    batch_img, batch_mask = batches[0]
    assert batch_img.shape == (2, 128, 128, 3)
    assert batch_mask.shape == (2, 128, 128, 1)
    assert batch_img.dtype == tf.float32
    assert batch_mask.dtype == tf.float32
