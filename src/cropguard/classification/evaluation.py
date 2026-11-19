"""
CropGuard AI - Classification Evaluation Suite.
Pure NumPy implementations of multi-class evaluation metrics, confusion matrix,
macro-averaged diagnostics, and hardware latency profiling.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf


def compute_binary_roc_auc(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Pure NumPy Wilcoxon-Mann-Whitney ROC-AUC calculation in O(N log N).
    
    Args:
        y_true_binary: Binary ground truth array (0 or 1).
        y_scores: Continuous confidence scores / predicted probabilities.
        
    Returns:
        Scalar ROC-AUC value in [0.0, 1.0].
    """
    pos_mask = (y_true_binary == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_scores)
    ranks = np.empty(len(y_scores), dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1, dtype=float)

    # Average ranks for ties
    sorted_scores = y_scores[order]
    unique_vals, inverse_idx, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_vals) < len(y_scores):
        tie_ranks = np.zeros(len(unique_vals), dtype=float)
        cur = 0
        for i, cnt in enumerate(counts):
            tie_ranks[i] = cur + (cnt + 1.0) / 2.0
            cur += cnt
        sorted_ranks = tie_ranks[inverse_idx]
        ranks[order] = sorted_ranks

    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_multiclass_roc_auc(
    y_true: Union[np.ndarray, List[int]],
    y_prob: np.ndarray,
    num_classes: int = 38,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes One-vs-Rest Multiclass ROC-AUC via pure NumPy Wilcoxon-Mann-Whitney rank sums.
    
    Args:
        y_true: Ground-truth integer class labels (shape [N]).
        y_prob: Predicted softmax probabilities (shape [N, K]).
        num_classes: Total number of classes (default: 38).
        class_names: Optional list of class names.
        
    Returns:
        Dictionary with macro_roc_auc and per_class_roc_auc.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    per_class_auc = {}
    valid_aucs = []

    for k in range(num_classes):
        name = class_names[k] if (class_names and k < len(class_names)) else f"class_{k}"
        y_bin = (y_true == k).astype(int)
        scores = y_prob[:, k] if k < y_prob.shape[1] else np.zeros(len(y_true))
        n_pos = int(np.sum(y_bin))
        if n_pos > 0 and n_pos < len(y_true):
            auc_k = compute_binary_roc_auc(y_bin, scores)
            per_class_auc[name] = auc_k
            valid_aucs.append(auc_k)
        else:
            per_class_auc[name] = 0.5

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return {
        "macro_roc_auc": macro_auc,
        "per_class_roc_auc": per_class_auc
    }


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 38,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class classification metrics:
    - Overall Accuracy: sum(TP) / N
    - Macro-Averaged Precision: (1/K) * sum(TP_k / (TP_k + FP_k))
    - Macro-Averaged Recall: (1/K) * sum(TP_k / (TP_k + FN_k))
    - Macro-Averaged F1-Score: (1/K) * sum(2 * P_k * R_k / (P_k + R_k))
    - One-vs-Rest Multiclass ROC-AUC (when probabilities provided)
    - Per-class Recall dictionary
    - Full K x K Confusion Matrix

    Pure NumPy implementation without external scikit-learn dependency.

    Args:
        y_true: Ground-truth integer class labels (shape [N]).
        y_pred: Predicted integer class labels (shape [N]) or softmax probabilities (shape [N, K]).
        y_prob: Optional predicted softmax probabilities (shape [N, K]).
        num_classes: Total number of target categories (38).
        class_names: Optional list of string class names.

    Returns:
        Dictionary containing scalar metrics, confusion matrix, and per-class breakdown.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)

    # If predicted probabilities passed in y_pred, extract y_prob
    if y_pred.ndim == 2:
        if y_prob is None:
            y_prob = y_pred.copy()
        y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.astype(int)

    # Construct confusion matrix: rows = ground truth, cols = predictions
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    tp = np.diag(cm).astype(float)
    fp = (np.sum(cm, axis=0) - tp).astype(float)
    fn = (np.sum(cm, axis=1) - tp).astype(float)

    total_samples = np.sum(cm)
    accuracy = float(np.sum(tp) / total_samples) if total_samples > 0 else 0.0

    # Class-wise metrics with zero-division safety
    precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp), where=(tp + fn) != 0)
    denom_f1 = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom_f1, out=np.zeros_like(precision), where=denom_f1 != 0)

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    per_class_recall_dict = {}
    for k in range(num_classes):
        name = class_names[k] if (class_names and k < len(class_names)) else f"class_{k}"
        per_class_recall_dict[name] = float(recall[k])

    result = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall_dict,
        "confusion_matrix": cm,
        "class_recalls_array": recall,
        "class_precisions_array": precision,
        "class_f1_array": f1,
        "total_samples": int(total_samples),
    }

    if y_prob is not None:
        roc_results = compute_multiclass_roc_auc(
            y_true, y_prob, num_classes=num_classes, class_names=class_names
        )
        result["macro_roc_auc"] = roc_results["macro_roc_auc"]
        result["per_class_roc_auc"] = roc_results["per_class_roc_auc"]

    return result


def measure_inference_latency(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int, int] = (1, 224, 224, 3),
    warmup_steps: int = 10,
    eval_steps: int = 50
) -> Dict[str, float]:
    """
    Profiles the empirical inference latency of a Keras model on the active accelerator.
    Performs initial warm-up to trigger JIT compilation and memory caching,
    then records high-resolution wall-clock timings.

    Args:
        model: Compiled or uncompiled tf.keras.Model.
        input_shape: 4D batch tensor shape (default [1, 224, 224, 3]).
        warmup_steps: Number of initial iterations to discard from timing.
        eval_steps: Number of iterations to measure.

    Returns:
        Dictionary with latency_mean_ms, latency_std_ms, and throughput_fps.
    """
    dummy_input = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, dtype=tf.float32)

    # Warm-up phase
    for _ in range(warmup_steps):
        _ = model(dummy_input, training=False)

    timings = []
    for _ in range(eval_steps):
        t0 = time.perf_counter()
        _ = model(dummy_input, training=False)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    latencies = np.array(timings)
    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return {
        "latency_mean_ms": mean_ms,
        "latency_std_ms": std_ms,
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "throughput_fps": fps,
        "batch_size": input_shape[0]
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    figsize: Tuple[int, int] = (16, 14),
    normalize: bool = True
) -> Path:
    """
    Renders and saves a high-resolution confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array of shape [K, K].
        class_names: List of K class names.
        output_path: Target path for the output PNG image.
        figsize: Figure dimensions in inches.
        normalize: Whether to normalize rows to show proportions.
        
    Returns:
        Path to saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.divide(cm_display, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Recall" if normalize else "Count", rotation=-90, va="bottom")

    num_classes = len(class_names)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_ylabel("True Pathological Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Pathological Class", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Validation Confusion Matrix ({'Normalized' if normalize else 'Raw Counts'})",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    plt.tight_layout()
    plt.savefig(str(out_p), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_p

