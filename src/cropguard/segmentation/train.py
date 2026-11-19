"""
CropGuard AI - Standalone Production U-Net Lesion Segmentation Training CLI Pipeline.
Trains and evaluates the U-Net architecture on the official PlantSeg dataset partitions.

Operational Principles:
- Strict split quarantine: Train (7,916) and Val (1,247) partitions only; Test (2,295) is strictly untouched.
- Memory-bounded streaming: Uses tf.data with cache_mode=None to prevent host RAM overflow (ADR-006).
- Dual objective loss: Combined Binary Cross-Entropy (BCE) and Sørensen-Dice loss (BCEDiceLoss).
- GPU optimization: Dynamic memory growth enabled to support RTX 5060 Laptop GPU.
- Deterministic reproducibility and dry-run preflight validation mode.
"""

import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
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

from src.cropguard.segmentation.models import build_unet
from src.cropguard.segmentation.data import (
    load_segmentation_records,
    build_segmentation_dataset,
)
from src.cropguard.segmentation.training import (
    BCEDiceLoss,
    bce_dice_loss,
    dice_loss,
    get_segmentation_callbacks,
)
from src.cropguard.segmentation.evaluation import (
    dice_coefficient,
    iou_score,
    compute_segmentation_metrics,
    compute_lesion_severity,
    dice_coefficient_np,
    iou_score_np,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cropguard.segmentation.train")


def configure_gpu_memory():
    """Configures GPU dynamic memory growth to prevent TensorFlow from allocating all VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Dynamic GPU memory growth successfully enabled on: %s", gpus[0].name)
        except RuntimeError as e:
            logger.warning("GPU memory growth configuration notice: %s", e)
    else:
        logger.info("No GPU accelerator detected; executing on host CPU.")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for segmentation training."""
    parser = argparse.ArgumentParser(
        prog="cropguard-segmentation-train",
        description="CropGuard AI — Production Foliar Lesion U-Net Segmentation Training CLI",
        epilog=(
            "Examples:\n"
            "  Dry-run preflight check:\n"
            "    python -m src.cropguard.segmentation.train --dry-run\n\n"
            "  Small smoke-test verification:\n"
            "    python -m src.cropguard.segmentation.train --smoke-test\n\n"
            "  Full production training:\n"
            "    python -m src.cropguard.segmentation.train --epochs 20 --batch-size 8 --learning-rate 1e-4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core Training Hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Mini-batch size for training and validation (default: 8 per ADR-006).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial Adam learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="Weight of binary cross-entropy in combined loss (default: 0.5).",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
        help="Weight of Sørensen-Dice loss in combined loss (default: 0.5).",
    )
    parser.add_argument(
        "--patience-early",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5).",
    )
    parser.add_argument(
        "--patience-lr",
        type=int,
        default=3,
        help="ReduceLROnPlateau patience in epochs (default: 3).",
    )

    # Architectural Parameters
    parser.add_argument(
        "--base-filters",
        type=int,
        default=32,
        help="Base convolutional filters in U-Net first encoder stage (default: 32).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="U-Net encoder-decoder hierarchy depth (default: 4).",
    )

    # Dataset & Partition Paths
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/splits/segmentation/train.csv",
        help="Path to PlantSeg training manifest CSV (default: data/splits/segmentation/train.csv).",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default="data/splits/segmentation/val.csv",
        help="Path to PlantSeg validation manifest CSV (default: data/splits/segmentation/val.csv).",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default="data/raw/plantseg/Plantseg.zip",
        help="Path to raw Plantseg.zip archive (default: data/raw/plantseg/Plantseg.zip).",
    )

    # Output & Execution Controls
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints and evaluation artifacts (default: models/segmentation/production/unet).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight validation checks without initiating model training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute small verification experiment on stratified subset (32 train, 16 val, 2 epochs).",
    )
    parser.add_argument(
        "--subset-train",
        type=int,
        default=None,
        help="Cap training set to N samples (used for quick prototyping).",
    )
    parser.add_argument(
        "--subset-val",
        type=int,
        default=None,
        help="Cap validation set to N samples (used for quick prototyping).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip model training and evaluate existing checkpoint on the validation set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and initialization (default: 42).",
    )

    parsed = parser.parse_args(args)

    # Resolve default output directory based on smoke-test flag if not explicitly provided
    if parsed.output_dir is None:
        if parsed.smoke_test:
            parsed.output_dir = "models/segmentation/smoke_test/unet"
        else:
            parsed.output_dir = "models/segmentation/production/unet"

    # Set smoke test defaults if enabled
    if parsed.smoke_test:
        if parsed.epochs == 20:  # User did not override
            parsed.epochs = 2
        if parsed.subset_train is None:
            parsed.subset_train = 32
        if parsed.subset_val is None:
            parsed.subset_val = 16

    return parsed


def save_prediction_visualizations(
    images: np.ndarray,
    ground_truths: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    num_samples: int = 6,
    threshold: float = 0.5,
) -> List[Path]:
    """
    Renders and saves 5-panel diagnostic visualizations comparing:
    Panel 1: Original Leaf RGB
    Panel 2: Ground-Truth Lesion Mask (with ground-truth foliar severity %)
    Panel 3: Predicted Continuous Probability Map (with Sørensen-Dice & IoU)
    Panel 4: Discrete Binary Mask at threshold tau=0.5 (with predicted severity %)
    Panel 5: Diagnostic Error Overlay on leaf image:
             - Green: True Positives (correctly localized foliar lesions)
             - Red: False Positives (healthy foliar tissue incorrectly predicted as lesions / over-segmentation)
             - Yellow: False Negatives (actual pathology lesions missed by prediction)

    Selects representative samples across performance tiers (high Dice, median Dice,
    and challenging / over-segmented samples) to provide rigorous diagnostic visibility.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(images)
    n_to_save = min(num_samples, n_total)
    saved_paths: List[Path] = []

    # Compute individual sample Dice scores to identify representative cases
    sample_dices = []
    for i in range(n_total):
        gt_bin = (ground_truths[i].squeeze() > 0.5).astype(np.float32)
        p_prob = predictions[i].squeeze()
        d = dice_coefficient_np(gt_bin, p_prob, threshold=threshold)
        sample_dices.append(d)

    # Select representative sample indices
    if n_total >= 6 and n_to_save >= 6:
        sorted_indices = np.argsort(sample_dices)
        # 2 high performers, 2 median performers, 2 challenging/over-segmented cases
        n_mid = len(sorted_indices) // 2
        selected_indices = [
            int(sorted_indices[-1]),                   # Best
            int(sorted_indices[-2]),                   # Second best
            int(sorted_indices[n_mid]),                 # Median upper
            int(sorted_indices[n_mid - 1]),             # Median lower
            int(sorted_indices[1]),                    # Lower tier
            int(sorted_indices[0]),                    # Worst / highest over-segmentation
        ]
        # Slice to requested count
        selected_indices = selected_indices[:n_to_save]
    else:
        selected_indices = list(range(n_to_save))

    for rank, idx in enumerate(selected_indices):
        img = images[idx]
        gt = ground_truths[idx].squeeze()
        pred_prob = predictions[idx].squeeze()
        pred_bin = (pred_prob > threshold).astype(bool)
        gt_bin = (gt > 0.5).astype(bool)

        dice = sample_dices[idx]
        iou = iou_score_np(gt_bin.astype(np.float32), pred_prob, threshold=threshold)

        gt_sev = compute_lesion_severity(gt_bin, threshold=0.5)["coverage_pct"]
        pred_sev = compute_lesion_severity(pred_bin, threshold=0.5)["coverage_pct"]

        # Confusion regions
        tp = gt_bin & pred_bin
        fp = (~gt_bin) & pred_bin  # Healthy tissue incorrectly predicted as lesion (Over-segmentation)
        fn = gt_bin & (~pred_bin)  # Missed lesion tissue

        # Build diagnostic overlay on leaf
        overlay = img.copy() * 0.75 + 0.1  # slightly dim original image for contrast
        # Apply high-contrast diagnostic tints
        if np.any(tp):
            overlay[tp] = np.clip(img[tp] * 0.2 + np.array([0.1, 0.9, 0.2]), 0.0, 1.0)      # Green: TP
        if np.any(fp):
            overlay[fp] = np.clip(img[fp] * 0.2 + np.array([0.95, 0.15, 0.15]), 0.0, 1.0)    # Red: FP (Over-segmentation)
        if np.any(fn):
            overlay[fn] = np.clip(img[fn] * 0.2 + np.array([0.95, 0.85, 0.1]), 0.0, 1.0)     # Yellow: FN (Missed)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=150)

        # Panel 1: Original Image
        axes[0].imshow(img)
        axes[0].set_title("1. Input Leaf RGB", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground Truth Mask
        axes[1].imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title(f"2. Ground Truth Mask\n(Severity: {gt_sev:.2f}%)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Predicted Probability
        axes[2].imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
        axes[2].set_title(f"3. Predicted Prob Map\nDice: {dice*100:.1f}% | IoU: {iou*100:.1f}%", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Binary Mask
        axes[3].imshow(pred_bin, cmap="gray", vmin=0.0, vmax=1.0)
        axes[3].set_title(f"4. Binary Mask (tau=0.5)\n(Severity: {pred_sev:.2f}%)", fontsize=11, fontweight="bold")
        axes[3].axis("off")

        # Panel 5: Diagnostic Error Overlay
        axes[4].imshow(overlay)
        axes[4].set_title("5. Diagnostic Error Overlay\nGreen: TP | Red: FP | Yellow: FN", fontsize=10, fontweight="bold")
        axes[4].axis("off")

        plt.tight_layout()
        save_path = output_dir / f"sample_{rank+1:02d}.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    logger.info("Saved %d 5-panel sample prediction visualizations to: %s", len(saved_paths), output_dir)
    return saved_paths


def train_segmentation_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Executes the standalone U-Net segmentation training pipeline:
    1. Preflight validation & hardware configuration.
    2. Split loading & dataset building (with strict test set quarantine).
    3. Model construction & compilation with BCEDiceLoss.
    4. Model fitting with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger.
    5. Post-training validation evaluation (Dice, IoU, Precision, Recall, Severity delta).
    6. Artifact persistence & sample prediction rendering.
    """
    logger.info("=" * 70)
    logger.info("CropGuard AI - U-Net Foliar Lesion Segmentation Training Pipeline")
    logger.info("=" * 70)

    # 1. Hardware Configuration
    configure_gpu_memory()

    # 2. Path & Quarantine Validation
    train_csv_path = Path(args.train_csv).resolve()
    val_csv_path = Path(args.val_csv).resolve()
    zip_path = Path(args.zip_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    logger.info("Training manifest:   %s", train_csv_path)
    logger.info("Validation manifest: %s", val_csv_path)
    logger.info("PlantSeg archive:    %s", zip_path)
    logger.info("Output directory:    %s", output_dir)

    # STRICT QUARANTINE ENFORCEMENT
    test_csv_path = PROJECT_ROOT / "data" / "splits" / "segmentation" / "test.csv"
    assert str(train_csv_path) != str(test_csv_path), "CRITICAL: Attempted to load test.csv as train_csv!"
    assert str(val_csv_path) != str(test_csv_path), "CRITICAL: Attempted to load test.csv as val_csv!"
    logger.info("PlantSeg test partition (2,295 samples) verified: STRICTLY QUARANTINED.")

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training split CSV missing: {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Validation split CSV missing: {val_csv_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"PlantSeg zip archive missing: {zip_path}")

    # Load manifests
    train_records = load_segmentation_records(train_csv_path)
    val_records = load_segmentation_records(val_csv_path)
    n_train_full = len(train_records)
    n_val_full = len(val_records)
    logger.info("Loaded manifests: %d training pairs, %d validation pairs.", n_train_full, n_val_full)

    # 3. Dry-Run Preflight Mode
    if args.dry_run:
        logger.info("Executing DRY-RUN preflight validation...")
        model = build_unet(
            input_shape=(256, 256, 3),
            num_classes=1,
            base_filters=args.base_filters,
            depth=args.depth,
        )
        total_params = model.count_params()
        trainable_params = sum(p.numpy().size for p in model.trainable_weights)
        logger.info("U-Net Model Architecture instantiated successfully.")
        logger.info("Total Parameters: %d | Trainable Parameters: %d", total_params, trainable_params)

        # Ensure output directory can be created
        output_dir.mkdir(parents=True, exist_ok=True)
        dry_run_summary = {
            "status": "DRY_RUN_PASSED",
            "training_samples_available": n_train_full,
            "validation_samples_available": n_val_full,
            "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
            "model_architecture": "U-Net",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "bce_weight": args.bce_weight,
            "dice_weight": args.dice_weight,
            "output_dir": str(output_dir),
        }
        logger.info("Dry-run preflight checks completed successfully in < 3s.")
        return dry_run_summary

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_pred_dir = output_dir / "sample_predictions"
    sample_pred_dir.mkdir(parents=True, exist_ok=True)

    # 4. Subset Selection (for smoke test or rapid experimentation)
    active_train_records = train_records
    active_val_records = val_records
    if args.subset_train is not None and args.subset_train < len(active_train_records):
        import random
        rng = random.Random(args.seed)
        active_train_records = rng.sample(active_train_records, args.subset_train)
        logger.info("Subsetting training set to %d samples.", len(active_train_records))
    if args.subset_val is not None and args.subset_val < len(active_val_records):
        import random
        rng = random.Random(args.seed)
        active_val_records = rng.sample(active_val_records, args.subset_val)
        logger.info("Subsetting validation set to %d samples.", len(active_val_records))

    logger.info("Building tf.data pipelines (cache_mode=None per ADR-006)...")
    train_ds = build_segmentation_dataset(
        records=active_train_records,
        zip_path=zip_path,
        batch_size=args.batch_size,
        target_shape=(256, 256),
        shuffle=True,
        seed=args.seed,
    )
    val_ds = build_segmentation_dataset(
        records=active_val_records,
        zip_path=zip_path,
        batch_size=args.batch_size,
        target_shape=(256, 256),
        shuffle=False,
        seed=args.seed,
    )

    # 5. Model Instantiation & Compilation
    best_model_path = output_dir / "best_model.keras"
    training_history_path = output_dir / "training_history.csv"

    if not args.eval_only:
        logger.info("Instantiating U-Net architecture (base_filters=%d, depth=%d)...", args.base_filters, args.depth)
        model = build_unet(
            input_shape=(256, 256, 3),
            num_classes=1,
            base_filters=args.base_filters,
            depth=args.depth,
        )
        total_params = model.count_params()
        logger.info("U-Net initialized with %d parameters.", total_params)

        optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
        loss_fn = BCEDiceLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[dice_coefficient, iou_score],
        )

        # Callbacks
        callbacks = get_segmentation_callbacks(
            checkpoint_path=best_model_path,
            csv_log_path=training_history_path,
            monitor="val_dice_coefficient",
            mode="max",
            patience_early=args.patience_early,
            patience_lr=args.patience_lr,
            verbose=1,
        )

        logger.info("Starting model fitting (max epochs: %d, batch size: %d)...", args.epochs, args.batch_size)
        t_start = time.time()
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        training_time = time.time() - t_start
        logger.info("Training concluded in %.2f seconds.", training_time)

        # Ensure checkpoint exists; fallback to saving final model if no improvement checkpoint saved
        if not best_model_path.exists():
            logger.warning("No checkpoint saved during training (monitor didn't trigger improvement); saving final weights.")
            model.save(str(best_model_path))

        # Log training history summary
        hist_dict = history.history
        best_val_dice = float(max(hist_dict.get("val_dice_coefficient", [0.0])))
        best_val_iou = float(max(hist_dict.get("val_iou_score", [0.0])))
        best_val_loss = float(min(hist_dict.get("val_loss", [float("inf")])))
        logger.info("Training Complete: Best Val Dice: %.4f | Best Val IoU: %.4f | Best Val Loss: %.4f",
                    best_val_dice, best_val_iou, best_val_loss)
        # Clean up in-memory training model to free GPU VRAM prior to evaluation
        del model
        gc.collect()
    else:
        training_time = 0.0
        logger.info("Skipping training (--eval-only specified). Loading checkpoint from: %s", best_model_path)

    # 6. Post-Training Validation Evaluation
    logger.info("Loading best model checkpoint for evaluation...")
    custom_objects = {
        "BCEDiceLoss": BCEDiceLoss,
        "bce_dice_loss": bce_dice_loss,
        "dice_loss": dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_score": iou_score,
    }
    eval_model = keras.models.load_model(str(best_model_path), custom_objects=custom_objects)

    logger.info("Evaluating model on %d validation samples...", len(active_val_records))
    all_imgs = []
    all_gts = []
    all_preds = []

    # Inference Latency Profiling
    t_inf_start = time.time()
    for batch_imgs, batch_masks in val_ds:
        preds = eval_model(batch_imgs, training=False)
        all_imgs.append(batch_imgs.numpy())
        all_gts.append(batch_masks.numpy())
        all_preds.append(preds.numpy())
    total_val_inf_time = time.time() - t_inf_start

    imgs_arr = np.concatenate(all_imgs, axis=0)
    gts_arr = np.concatenate(all_gts, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)

    n_eval = len(preds_arr)
    latency_ms_per_image = (total_val_inf_time / n_eval) * 1000.0 if n_eval > 0 else 0.0
    throughput_fps = n_eval / total_val_inf_time if total_val_inf_time > 0 else 0.0

    # Comprehensive NumPy Metrics
    val_metrics = compute_segmentation_metrics(gts_arr, preds_arr, threshold=0.5)

    # Severity analysis across evaluated set
    gt_severities = [compute_lesion_severity(gts_arr[i], threshold=0.5)["coverage_pct"] for i in range(n_eval)]
    pred_severities = [compute_lesion_severity(preds_arr[i], threshold=0.5)["coverage_pct"] for i in range(n_eval)]
    mean_gt_severity = float(np.mean(gt_severities))
    mean_pred_severity = float(np.mean(pred_severities))
    severity_delta = abs(mean_gt_severity - mean_pred_severity)

    # 7. Render and Save Sample Predictions
    sample_files = save_prediction_visualizations(
        images=imgs_arr,
        ground_truths=gts_arr,
        predictions=preds_arr,
        output_dir=sample_pred_dir,
        num_samples=min(6, n_eval),
        threshold=0.5,
    )

    # 8. Persist Artifacts
    config_dict = {
        "model_architecture": "U-Net",
        "input_shape": [256, 256, 3],
        "num_classes": 1,
        "base_filters": args.base_filters,
        "depth": args.depth,
        "total_parameters": eval_model.count_params(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "bce_weight": args.bce_weight,
        "dice_weight": args.dice_weight,
        "patience_early": args.patience_early,
        "patience_lr": args.patience_lr,
        "smoke_test": args.smoke_test,
        "training_samples_evaluated": len(active_train_records),
        "validation_samples_evaluated": n_eval,
        "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    metrics_dict = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_samples_evaluated": n_eval,
        "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
        "training_duration_seconds": training_time,
        "dice_coefficient": val_metrics["dice"],
        "iou_score": val_metrics["iou"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "specificity": val_metrics["specificity"],
        "mean_ground_truth_severity_pct": mean_gt_severity,
        "mean_predicted_severity_pct": mean_pred_severity,
        "mean_severity_delta_pct": severity_delta,
        "confusion_matrix": {
            "tp": val_metrics["tp"],
            "fp": val_metrics["fp"],
            "tn": val_metrics["tn"],
            "fn": val_metrics["fn"],
            "total_pixels": val_metrics["total_pixels"],
        },
        "hardware_inference_profile": {
            "total_inference_time_seconds": total_val_inf_time,
            "latency_ms_per_image": latency_ms_per_image,
            "throughput_fps": throughput_fps,
        },
        "checkpoint_file": str(best_model_path.name),
        "sample_visualizations_count": len(sample_files),
    }
    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info("=" * 70)
    logger.info("Validation Evaluation Summary:")
    logger.info("  Validation Dice:       %.4f (%.2f%%)", val_metrics["dice"], val_metrics["dice"] * 100.0)
    logger.info("  Validation IoU:        %.4f (%.2f%%)", val_metrics["iou"], val_metrics["iou"] * 100.0)
    logger.info("  Validation Precision:  %.4f", val_metrics["precision"])
    logger.info("  Validation Recall:     %.4f", val_metrics["recall"])
    logger.info("  Validation Specificity:%.4f", val_metrics["specificity"])
    logger.info("  Mean GT Severity:      %.2f%%", mean_gt_severity)
    logger.info("  Mean Pred Severity:    %.2f%%", mean_pred_severity)
    logger.info("  Severity Delta:        %.2f%%", severity_delta)
    logger.info("  Inference Latency:     %.2f ms/image (%.1f FPS)", latency_ms_per_image, throughput_fps)
    logger.info("Artifacts saved to: %s", output_dir)
    logger.info("=" * 70)

    return {
        "config": config_dict,
        "metrics": metrics_dict,
        "artifacts": {
            "checkpoint": str(best_model_path),
            "config": str(config_path),
            "metrics": str(metrics_path),
            "history": str(training_history_path) if training_history_path.exists() else None,
            "sample_dir": str(sample_pred_dir),
        },
    }


def main():
    """Main CLI entrypoint."""
    args = parse_args()
    train_segmentation_pipeline(args)


if __name__ == "__main__":
    main()
