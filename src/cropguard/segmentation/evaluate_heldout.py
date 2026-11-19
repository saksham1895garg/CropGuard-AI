"""
CropGuard AI - Final Held-Out Test Evaluation CLI for Foliar Lesion Segmentation.
Evaluates the frozen production U-Net segmentation checkpoint strictly against
the official held-out PlantSeg test partition (2,295 image-mask pairs).
Reuses existing segmentation models, tf.data loader, and evaluation metrics without retraining.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.segmentation.data import (
    load_segmentation_records,
    build_segmentation_dataset,
)
from src.cropguard.segmentation.training import (
    BCEDiceLoss,
    bce_dice_loss,
    dice_loss,
)
from src.cropguard.segmentation.evaluation import (
    dice_coefficient,
    iou_score,
    dice_coefficient_np,
    iou_score_np,
    compute_segmentation_metrics,
    compute_lesion_severity,
)
from src.cropguard.segmentation.train import save_prediction_visualizations

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cropguard.segmentation.evaluate_heldout")


def configure_gpu_memory():
    """Configures GPU memory growth to prevent TensorFlow from allocating all VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Dynamic GPU memory growth enabled on: %s", gpus[0].name)
        except RuntimeError as e:
            logger.warning("GPU memory growth configuration notice: %s", e)
    else:
        logger.info("No GPU accelerator detected; executing on host CPU.")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses CLI arguments for held-out test evaluation."""
    parser = argparse.ArgumentParser(
        prog="cropguard-segmentation-test-eval",
        description="CropGuard AI — Final Held-Out Test Evaluation for U-Net Lesion Segmentation",
        epilog=(
            "Example:\n"
            "  python src/cropguard/segmentation/evaluate_heldout.py \\\n"
            "    --model-path models/segmentation/production/unet/best_model.keras \\\n"
            "    --test-csv data/splits/segmentation/test.csv \\\n"
            "    --output-dir reports/final_evaluation\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/segmentation/production/unet/best_model.keras",
        help="Path to frozen Keras production model checkpoint (.keras).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/splits/segmentation/test.csv",
        help="Path to PlantSeg held-out test partition CSV (default: data/splits/segmentation/test.csv).",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="data/raw/plantseg/Plantseg.zip",
        help="Path to raw Plantseg.zip archive (default: data/raw/plantseg/Plantseg.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/final_evaluation",
        help="Directory to persist test evaluation metrics and reports (default: reports/final_evaluation).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for test evaluation (default: 8 per ADR-006).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation binarization threshold (default: 0.5).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of representative diagnostic sample visualizations to save (default: 6).",
    )
    return parser.parse_args(args)


def run_heldout_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executes the held-out test evaluation on the frozen production U-Net model:
    1. Loads the frozen checkpoint.
    2. Streams all 2,295 test pairs via memory-safe tf.data.
    3. Evaluates overall and per-category segmentation metrics.
    4. Generates representative 5-panel visual comparisons.
    5. Persists structured JSON and CSV reports.
    """
    logger.info("=" * 70)
    logger.info("CropGuard AI - U-Net Held-Out Test Evaluation Pipeline")
    logger.info("=" * 70)

    configure_gpu_memory()

    model_path = Path(args.model_path).resolve()
    test_csv_path = Path(args.test_csv).resolve()
    zip_path = Path(args.zip_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "unet_sample_predictions"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_csv_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"PlantSeg zip archive not found: {zip_path}")

    # Load test manifest
    records = load_segmentation_records(test_csv_path)
    n_test = len(records)
    logger.info("Loaded %d held-out test records from: %s", n_test, test_csv_path)

    # Load model
    logger.info("Loading frozen production U-Net checkpoint from: %s", model_path)
    custom_objects = {
        "BCEDiceLoss": BCEDiceLoss,
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
    }
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    total_params = model.count_params()
    logger.info("Model loaded successfully: %d parameters. Input: %s, Output: %s",
                total_params, model.input_shape, model.output_shape)

    # Build tf.data pipeline
    logger.info("Building tf.data input pipeline for test partition...")
    test_ds = build_segmentation_dataset(
        records=records,
        zip_path=zip_path,
        batch_size=args.batch_size,
        target_shape=(256, 256),
        shuffle=False,
    )

    # Inference loop with latency profiling
    logger.info("Running batched inference across all %d test samples...", n_test)
    all_imgs = []
    all_gts = []
    all_preds = []

    t_start = time.time()
    for batch_imgs, batch_masks in test_ds:
        preds = model(batch_imgs, training=False)
        all_imgs.append(batch_imgs.numpy())
        all_gts.append(batch_masks.numpy())
        all_preds.append(preds.numpy())
    total_eval_time = time.time() - t_start

    imgs_arr = np.concatenate(all_imgs, axis=0)
    gts_arr = np.concatenate(all_gts, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)

    latency_ms_per_image = (total_eval_time / n_test) * 1000.0
    throughput_fps = n_test / total_eval_time

    logger.info("Inference completed in %.2f seconds. Latency: %.2f ms/image (%.2f FPS).",
                total_eval_time, latency_ms_per_image, throughput_fps)

    # Overall dataset-level metrics
    logger.info("Computing global pixel confusion metrics (threshold=%.2f)...", args.threshold)
    global_metrics = compute_segmentation_metrics(gts_arr, preds_arr, threshold=args.threshold)

    # Sample-level severities
    gt_severities = [compute_lesion_severity(gts_arr[i], threshold=args.threshold)["coverage_pct"] for i in range(n_test)]
    pred_severities = [compute_lesion_severity(preds_arr[i], threshold=args.threshold)["coverage_pct"] for i in range(n_test)]
    mean_gt_sev = float(np.mean(gt_severities))
    mean_pred_sev = float(np.mean(pred_severities))
    mean_sev_delta = float(np.mean(np.abs(np.array(gt_severities) - np.array(pred_severities))))

    # Per-category breakdown
    logger.info("Computing per-category pathology breakdown across test classes...")
    categories: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(records):
        plant = rec.get("plant", "Unknown")
        disease = rec.get("disease", "Unknown")
        cat_key = f"{plant} - {disease}"
        if cat_key not in categories:
            categories[cat_key] = {
                "plant": plant,
                "disease": disease,
                "indices": [],
            }
        categories[cat_key]["indices"].append(i)

    per_class_rows = []
    for cat_key, cat_data in sorted(categories.items()):
        idxs = cat_data["indices"]
        sub_gt = gts_arr[idxs]
        sub_pred = preds_arr[idxs]
        sub_metrics = compute_segmentation_metrics(sub_gt, sub_pred, threshold=args.threshold)
        sub_gt_sev = float(np.mean([gt_severities[k] for k in idxs]))
        sub_pred_sev = float(np.mean([pred_severities[k] for k in idxs]))
        sub_delta = abs(sub_gt_sev - sub_pred_sev)

        row = {
            "plant": cat_data["plant"],
            "disease": cat_data["disease"],
            "test_samples": len(idxs),
            "dice": round(sub_metrics["dice"], 4),
            "iou": round(sub_metrics["iou"], 4),
            "precision": round(sub_metrics["precision"], 4),
            "recall": round(sub_metrics["recall"], 4),
            "specificity": round(sub_metrics["specificity"], 4),
            "mean_gt_severity_pct": round(sub_gt_sev, 2),
            "mean_pred_severity_pct": round(sub_pred_sev, 2),
            "severity_delta_pct": round(sub_delta, 2),
            "tp": sub_metrics["tp"],
            "fp": sub_metrics["fp"],
            "tn": sub_metrics["tn"],
            "fn": sub_metrics["fn"],
        }
        per_class_rows.append(row)

    # Save per-class CSV
    per_class_csv_path = output_dir / "unet_test_per_class_metrics.csv"
    with open(per_class_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "plant", "disease", "test_samples", "dice", "iou", "precision",
            "recall", "specificity", "mean_gt_severity_pct", "mean_pred_severity_pct",
            "severity_delta_pct", "tp", "fp", "tn", "fn"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)
    logger.info("Saved per-class metrics to: %s", per_class_csv_path)

    # Generate 5-panel sample visualizations
    logger.info("Generating representative 5-panel test visualizations...")
    saved_samples = save_prediction_visualizations(
        images=imgs_arr,
        ground_truths=gts_arr,
        predictions=preds_arr,
        output_dir=sample_dir,
        num_samples=args.num_samples,
        threshold=args.threshold,
    )

    # Save summary JSON
    summary_report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "U-Net",
        "model_path": str(model_path),
        "total_parameters": total_params,
        "test_dataset": "PlantSeg (held-out official test partition)",
        "test_csv_path": str(test_csv_path),
        "test_samples_evaluated": n_test,
        "distinct_classes_evaluated": len(per_class_rows),
        "threshold": args.threshold,
        "dice_coefficient": global_metrics["dice"],
        "iou_score": global_metrics["iou"],
        "precision": global_metrics["precision"],
        "recall": global_metrics["recall"],
        "specificity": global_metrics["specificity"],
        "mean_ground_truth_severity_pct": mean_gt_sev,
        "mean_predicted_severity_pct": mean_pred_sev,
        "mean_severity_delta_pct": mean_sev_delta,
        "confusion_matrix": {
            "tp": global_metrics["tp"],
            "fp": global_metrics["fp"],
            "tn": global_metrics["tn"],
            "fn": global_metrics["fn"],
            "total_pixels": global_metrics["total_pixels"],
        },
        "hardware_inference_profile": {
            "total_eval_time_seconds": total_eval_time,
            "latency_ms_per_image": latency_ms_per_image,
            "throughput_fps": throughput_fps,
            "batch_size": args.batch_size,
        },
        "sample_visualizations_count": len(saved_samples),
        "per_class_metrics_file": str(per_class_csv_path.name),
    }

    metrics_json_path = output_dir / "unet_test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Saved test evaluation metrics to: %s", metrics_json_path)

    logger.info("=" * 70)
    logger.info("Final U-Net Test Performance Summary (%d samples):", n_test)
    logger.info("  Held-Out Test Dice:        %.4f (%.2f%%)", global_metrics["dice"], global_metrics["dice"] * 100.0)
    logger.info("  Held-Out Test IoU:         %.4f (%.2f%%)", global_metrics["iou"], global_metrics["iou"] * 100.0)
    logger.info("  Held-Out Test Precision:   %.4f", global_metrics["precision"])
    logger.info("  Held-Out Test Recall:      %.4f", global_metrics["recall"])
    logger.info("  Held-Out Test Specificity: %.4f", global_metrics["specificity"])
    logger.info("  Mean GT Lesion Severity:   %.2f%%", mean_gt_sev)
    logger.info("  Mean Pred Lesion Severity: %.2f%%", mean_pred_sev)
    logger.info("  Severity Delta:            %.2f%%", mean_sev_delta)
    logger.info("  Hardware Latency:          %.2f ms/image (%.2f FPS)", latency_ms_per_image, throughput_fps)
    logger.info("=" * 70)

    return summary_report


def main():
    args = parse_args()
    run_heldout_evaluation(args)


if __name__ == "__main__":
    main()
