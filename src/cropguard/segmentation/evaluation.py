"""
CropGuard AI - Foliar Lesion Segmentation Evaluation & Metrics
Implements Keras-compatible metrics for training and pure NumPy metrics
for evaluation and test benchmarking:
- Dice Similarity Coefficient (F1-score for binary masks)
- Intersection over Union (mIoU / Jaccard Index)
- Confusion metrics (Precision, Recall, Specificity)
- Foliar Pathology Severity Index (percentage of necrotic/chlorotic lesion area)
"""

from typing import Dict, Any, Union
import numpy as np
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="cropguard.segmentation")
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Sørensen-Dice Coefficient for Keras training / evaluation:
    Dice = (2 * sum(y_true * y_pred) + smooth) / (sum(y_true) + sum(y_pred) + smooth)
    Evaluates probability output y_pred directly for differentiable tracking.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = list(range(1, tf.rank(y_true)))
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)


@keras.saving.register_keras_serializable(package="cropguard.segmentation")
def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> tf.Tensor:
    """
    Intersection over Union (Jaccard Index) for Keras training / evaluation:
    IoU = (sum(y_true * y_bin) + smooth) / (sum(y_true) + sum(y_bin) - sum(y_true * y_bin) + smooth)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_bin = tf.cast(y_pred > threshold, tf.float32)

    axes = list(range(1, tf.rank(y_true)))
    intersection = tf.reduce_sum(y_true * y_bin, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_bin, axis=axes) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def dice_coefficient_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Pure NumPy Sørensen-Dice coefficient for binarized predictions.
    """
    y_t = (y_true > threshold).astype(np.float32)
    y_p = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_t * y_p)
    denominator = np.sum(y_t) + np.sum(y_p)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice)


def iou_score_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Pure NumPy Intersection over Union (Jaccard Index) for binarized predictions.
    """
    y_t = (y_true > threshold).astype(np.float32)
    y_p = (y_pred > threshold).astype(np.float32)

    intersection = np.sum(y_t * y_p)
    union = np.sum(y_t) + np.sum(y_p) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> Dict[str, Any]:
    """
    Computes comprehensive pure NumPy binary segmentation metrics:
    - Dice similarity coefficient (F1 score)
    - Intersection over Union (mIoU / Jaccard)
    - Precision (positive predictive value)
    - Recall (sensitivity)
    - Specificity (true negative rate)
    - Confusion matrix counts (TP, FP, TN, FN)
    """
    y_t = (y_true > 0.5).astype(bool)
    y_p = (y_pred > threshold).astype(bool)

    tp = int(np.sum(y_t & y_p))
    fp = int(np.sum(~y_t & y_p))
    tn = int(np.sum(~y_t & ~y_p))
    fn = int(np.sum(y_t & ~y_p))

    dice = (2.0 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_pixels": int(y_true.size)
    }


def compute_lesion_severity(
    mask: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Union[int, float]]:
    """
    Computes foliar lesion severity metrics from a predicted or ground-truth mask.
    Returns:
        - lesion_pixels: count of positive lesion pixels
        - total_pixels: total image pixel count
        - coverage_pct: percentage of foliar area infected (0.0% to 100.0%)
    """
    bin_m = (mask > threshold).astype(bool)
    lesion_pixels = int(np.count_nonzero(bin_m))
    total_pixels = int(mask.size)
    coverage_pct = (lesion_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    return {
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 4)
    }
