"""
CropGuard AI - Production Classification Training CLI Pipeline.
Executes scalable, memory-safe training for foliar pathology classification
on the leak-free PlantVillage group-aware partitions.

Supported Architectures:
- custom_cnn: 4-stage convolutional neural network trained from scratch.
- mobilenet_v2: Depthwise separable convolutional network with ImageNet transfer learning.
- resnet50: Deep residual network with ImageNet transfer learning.

Two-Phase Training Strategy:
- Phase 1 (Feature Extraction): Frozen backbone, trains classification head (lr=1e-3).
- Phase 2 (Fine-Tuning): Unfreezes top 20-30 backbone layers, keeps BatchNorm frozen (lr=1e-5).

Operational Principles:
- Group-aware partition isolation: Train (35,221), Val (8,085), Test (10,999 strictly held out).
- Strict separation of production artifacts from smoke-test artifacts.
- Deterministic reproducibility and dry-run validation mode.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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
from src.cropguard.classification.training import (
    compile_classifier,
    get_training_callbacks,
)
from src.cropguard.classification.evaluation import (
    compute_classification_metrics,
    measure_inference_latency,
    save_confusion_matrix_plot,
)
from src.cropguard.data.loader import build_classification_dataset, load_dataset_records

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cropguard.classification.train")


def configure_gpu_memory():
    """Configures GPU memory growth to prevent TensorFlow from allocating all VRAM."""
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
    """Parses command-line arguments for production training."""
    parser = argparse.ArgumentParser(
        prog="cropguard-train",
        description="CropGuard AI — Production Folier Pathology Classifier Training CLI",
        epilog=(
            "Examples:\n"
            "  Dry-run check:\n"
            "    python -m src.cropguard.classification.train --model mobilenet_v2 --dry-run\n\n"
            "  Train Custom CNN:\n"
            "    python -m src.cropguard.classification.train --model custom_cnn --epochs 10 --batch-size 32\n\n"
            "  Train MobileNetV2 with Fine-Tuning:\n"
            "    python -m src.cropguard.classification.train --model mobilenet_v2 --epochs 10 "
            "--fine-tune-layers 25 --fine-tune-epochs 5 --batch-size 32\n\n"
            "  Train ResNet50:\n"
            "    python -m src.cropguard.classification.train --model resnet50 --epochs 10 "
            "--fine-tune-layers 30 --fine-tune-epochs 5 --batch-size 16\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["custom_cnn", "mobilenet_v2", "resnet50"],
        help="Target neural network architecture to build and train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for Phase 1 feature extraction / scratch training (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for training and validation (default: 32).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for Phase 1 Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic initialization and data shuffling (default: 42).",
    )
    parser.add_argument(
        "--fine-tune-layers",
        type=int,
        default=25,
        help="Number of top backbone layers to unfreeze in Phase 2 for transfer learning (default: 25).",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs for Phase 2 (default: 5).",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Reduced learning rate for Phase 2 fine-tuning (default: 1e-5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/classification/production",
        help="Destination directory for production checkpoints and evaluation reports.",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="none",
        choices=["none", "disk", "memory"],
        help="Caching policy for tf.data ('none': safe streaming, 'disk': disk-backed, 'memory': RAM).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional label smoothing epsilon for crossentropy loss (default: 0.0).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Patience epochs for EarlyStopping before halting (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configurations, dataset partitions, model setup, and paths without training.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run validation evaluation on existing best_model.keras in output directory.",
    )

    return parser.parse_args(args)


def validate_dataset_partitions() -> Tuple[Path, Path, Path, int, int, int]:
    """
    Validates the existence and isolation of group-aware dataset partitions.
    Guarantees that test partition is present for future held-out benchmarking
    while ensuring it is never touched during training or validation.
    """
    splits_dir = PROJECT_ROOT / "data" / "splits" / "classification_group_aware"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Required group-aware split file missing: {p}")

    train_fps, train_lbls, _ = load_dataset_records(train_csv)
    val_fps, val_lbls, _ = load_dataset_records(val_csv)
    test_fps, test_lbls, _ = load_dataset_records(test_csv)

    n_train, n_val, n_test = len(train_fps), len(val_fps), len(test_fps)

    # Disjointness assertion
    train_set, val_set, test_set = set(train_fps), set(val_fps), set(test_fps)
    assert len(train_set & val_set) == 0, "Train and Val partitions have overlapping samples!"
    assert len(train_set & test_set) == 0, "Train and Test partitions have overlapping samples!"
    assert len(val_set & test_set) == 0, "Val and Test partitions have overlapping samples!"

    logger.info("Dataset partitions validated: Train=%d, Val=%d, Test=%d (Held out)", n_train, n_val, n_test)
    return train_csv, val_csv, test_csv, n_train, n_val, n_test


def build_model_by_name(
    model_name: str,
    num_classes: int = 38
) -> Tuple[tf.keras.Model, Optional[tf.keras.Model]]:
    """Builds the requested neural network model and returns (model, backbone_or_None)."""
    if model_name == "custom_cnn":
        model = build_custom_foliar_cnn(input_shape=(224, 224, 3), num_classes=num_classes)
        return model, None
    elif model_name == "mobilenet_v2":
        model, backbone = build_mobilenet_v2_classifier(
            input_shape=(224, 224, 3),
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True,
        )
        return model, backbone
    elif model_name == "resnet50":
        model, backbone = build_resnet50_classifier(
            input_shape=(224, 224, 3),
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True,
        )
        return model, backbone
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def execute_dry_run(
    args: argparse.Namespace,
    model_dir: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    n_train: int,
    n_val: int,
    n_test: int,
    classes: List[str],
    class_weights: Dict[int, float]
) -> Dict[str, Any]:
    """Executes a non-training dry-run validation and writes config.json."""
    logger.info("=== EXECUTING DRY-RUN CONFIGURATION VALIDATION ===")

    # 1. Model architecture verification
    model, backbone = build_model_by_name(args.model, num_classes=len(classes))
    total_params = model.count_params()
    trainable_params_p1 = sum(p.numpy().size for p in model.trainable_weights)
    non_trainable_params_p1 = sum(p.numpy().size for p in model.non_trainable_weights)

    # 2. Fine-tuning capability verification
    trainable_params_p2 = 0
    if backbone is not None and args.fine_tune_layers > 0:
        # Simulate unfreezing top layers
        unfrozen_weights = set_backbone_trainable_layers(backbone, args.fine_tune_layers)
        trainable_params_p2 = sum(p.numpy().size for p in model.trainable_weights)

    # 3. Preprocessing assignment
    preprocess_fn = get_model_preprocessing(args.model)
    test_tensor = tf.constant([[[[0.0, 127.5, 255.0]]]], dtype=tf.float32)
    sample_preprocessed = preprocess_fn(test_tensor).numpy()

    # 4. Prepare output structure
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tensorboard" / "phase1").mkdir(parents=True, exist_ok=True)
    if backbone is not None:
        (model_dir / "tensorboard" / "phase2").mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dry_run": True,
        "status": "VALIDATED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_architecture": args.model,
        "classes_count": len(classes),
        "total_parameters": total_params,
        "phase1_trainable_parameters": trainable_params_p1,
        "phase1_non_trainable_parameters": non_trainable_params_p1,
        "phase2_trainable_parameters": trainable_params_p2,
        "supports_fine_tuning": backbone is not None,
        "fine_tune_layers_configured": args.fine_tune_layers if backbone is not None else 0,
        "fine_tune_epochs_configured": args.fine_tune_epochs if backbone is not None else 0,
        "fine_tune_lr_configured": args.fine_tune_lr if backbone is not None else None,
        "phase1_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "random_seed": args.seed,
        "cache_mode": args.cache_mode,
        "label_smoothing": args.label_smoothing,
        "dataset_partitions": {
            "train_csv": str(train_csv),
            "train_samples": n_train,
            "val_csv": str(val_csv),
            "val_samples": n_val,
            "test_csv": str(test_csv),
            "test_samples": n_test,
            "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
        },
        "class_weights_sample": {
            "min_weight": float(min(class_weights.values())),
            "max_weight": float(max(class_weights.values())),
            "conservation_sum": float(sum(class_weights[k] for k in range(len(classes)))),
        },
        "output_directory": str(model_dir),
        "target_checkpoints": {
            "best_overall_model": str(model_dir / "best_model.keras"),
            "phase1_best_model": str(model_dir / "best_model_phase1.keras"),
            "phase2_best_model": str(model_dir / "best_model_phase2.keras") if backbone is not None else None,
            "phase1_history_csv": str(model_dir / "history_phase1.csv"),
            "phase2_history_csv": str(model_dir / "history_phase2.csv") if backbone is not None else None,
            "val_evaluation_report": str(model_dir / "val_evaluation_report.json"),
            "confusion_matrix_png": str(model_dir / "confusion_matrix.png"),
        },
        "verified_sample_preprocessed_range": [float(np.min(sample_preprocessed)), float(np.max(sample_preprocessed))],
    }

    config_path = model_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dry-run configuration JSON persisted to: %s", config_path)
    print("\n" + "=" * 70)
    print(f"CROPGUARD AI — DRY-RUN VALIDATION SUCCESSFUL [{args.model.upper()}]")
    print("=" * 70)
    print(f"Architecture         : {args.model}")
    print(f"Total Parameters     : {total_params:,d}")
    print(f"Phase 1 Trainable    : {trainable_params_p1:,d}")
    if backbone is not None:
        print(f"Phase 2 Trainable    : {trainable_params_p2:,d} (Top {args.fine_tune_layers} layers)")
    print(f"Training Samples     : {n_train:,d} (from {train_csv.name})")
    print(f"Validation Samples   : {n_val:,d} (from {val_csv.name})")
    print(f"Test Samples         : {n_test:,d} (STRICTLY HELD OUT — {test_csv.name})")
    print(f"Output Directory     : {model_dir}")
    print(f"Config Record        : {config_path}")
    print("=" * 70)
    print("Status: READY FOR PRODUCTION EXECUTION. Test split isolation 100% verified.\n")
    return summary


def train_production_pipeline(args: argparse.Namespace):
    """Executes the full two-phase production training and validation evaluation workflow."""
    configure_gpu_memory()

    # 1. Deterministic seeding
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # 2. Dataset validation
    train_csv, val_csv, test_csv, n_train, n_val, n_test = validate_dataset_partitions()
    classes, class_to_idx, idx_to_class = get_plantvillage_classes()
    class_weights = compute_class_weights(train_csv, class_to_idx)

    # 3. Directory structure setup
    base_out = Path(args.output_dir)
    if not base_out.is_absolute():
        base_out = PROJECT_ROOT / base_out
    model_dir = base_out / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    # 4. Dry-run mode short-circuit
    if args.dry_run:
        execute_dry_run(
            args=args,
            model_dir=model_dir,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            classes=classes,
            class_weights=class_weights,
        )
        return

    # 5. Build streaming data pipelines (Test partition is NEVER loaded)
    preprocess_fn = get_model_preprocessing(args.model)
    logger.info("Initializing high-throughput streaming tf.data pipelines...")
    ds_train = build_classification_dataset(
        csv_path=train_csv,
        batch_size=args.batch_size,
        shuffle=True,
        cache_mode=None if args.cache_mode == "none" else args.cache_mode,
        preprocess_fn=preprocess_fn,
        seed=args.seed,
    )
    ds_val = build_classification_dataset(
        csv_path=val_csv,
        batch_size=args.batch_size,
        shuffle=False,
        cache_mode=None if args.cache_mode == "none" else args.cache_mode,
        preprocess_fn=preprocess_fn,
        seed=args.seed,
    )

    # 6. Training Execution (Bypassed if --eval-only is passed)
    final_best_model_path = model_dir / "best_model.keras"
    if args.eval_only:
        if not final_best_model_path.exists():
            raise FileNotFoundError(f"Cannot run in --eval-only mode: checkpoint not found at {final_best_model_path}")
        logger.info("Executing in --eval-only mode with existing model checkpoint: %s", final_best_model_path)
        p1_time = 0.0
        p2_time = 0.0
        best_ckpt_path = final_best_model_path
    else:
        # Instantiate model
        model, backbone = build_model_by_name(args.model, num_classes=len(classes))
        logger.info("Model '%s' instantiated with %s parameters.", args.model, f"{model.count_params():,d}")

        # =========================================================================
        # PHASE 1: FEATURE EXTRACTION / INITIAL TRAINING
        # =========================================================================
        logger.info("Starting Phase 1 Training (Epochs: %d, LR: %e)...", args.epochs, args.learning_rate)
        compile_classifier(model, learning_rate=args.learning_rate, label_smoothing=args.label_smoothing)

        phase1_ckpt = model_dir / "best_model_phase1.keras"
        phase1_log = model_dir / "history_phase1.csv"
        phase1_tb = model_dir / "tensorboard" / "phase1"

        callbacks_p1 = get_training_callbacks(
            checkpoint_path=phase1_ckpt,
            log_file=phase1_log,
            tensorboard_dir=phase1_tb,
            patience=args.patience,
        )

        t0 = time.time()
        history_p1 = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=callbacks_p1,
            verbose=1,
        )
        p1_time = time.time() - t0
        logger.info("Phase 1 completed in %.2f seconds.", p1_time)

        # Safe memory cleanup
        gc.collect()

        best_ckpt_path = phase1_ckpt
        p1_best_val_loss = float(min(history_p1.history.get("val_loss", [float("inf")])))
        logger.info("Phase 1 Best Validation Loss: %.4f", p1_best_val_loss)

        # =========================================================================
        # PHASE 2: TRANSFER LEARNING FINE-TUNING (MobileNetV2 & ResNet50 only)
        # =========================================================================
        p2_time = 0.0
        if backbone is not None and args.fine_tune_layers > 0 and args.fine_tune_epochs > 0:
            logger.info(
                "Starting Phase 2 Fine-Tuning (Unfreezing top %d layers, Epochs: %d, LR: %e)...",
                args.fine_tune_layers,
                args.fine_tune_epochs,
                args.fine_tune_lr,
            )

            # Unfreeze top N eligible layers (keeping BatchNorm frozen)
            set_backbone_trainable_layers(backbone, num_layers_to_unfreeze=args.fine_tune_layers)
            compile_classifier(model, learning_rate=args.fine_tune_lr, label_smoothing=args.label_smoothing)

            phase2_ckpt = model_dir / "best_model_phase2.keras"
            phase2_log = model_dir / "history_phase2.csv"
            phase2_tb = model_dir / "tensorboard" / "phase2"

            callbacks_p2 = get_training_callbacks(
                checkpoint_path=phase2_ckpt,
                log_file=phase2_log,
                tensorboard_dir=phase2_tb,
                patience=args.patience,
            )

            t2_start = time.time()
            history_p2 = model.fit(
                ds_train,
                validation_data=ds_val,
                epochs=args.fine_tune_epochs,
                class_weight=class_weights,
                callbacks=callbacks_p2,
                verbose=1,
            )
            p2_time = time.time() - t2_start
            logger.info("Phase 2 fine-tuning completed in %.2f seconds.", p2_time)
            gc.collect()

            p2_best_val_loss = float(min(history_p2.history.get("val_loss", [float("inf")])))
            logger.info("Phase 2 Best Validation Loss: %.4f", p2_best_val_loss)

            if p2_best_val_loss < p1_best_val_loss and phase2_ckpt.exists():
                best_ckpt_path = phase2_ckpt
                logger.info("Phase 2 achieved superior validation loss (%.4f vs %.4f); selecting Phase 2 model.", p2_best_val_loss, p1_best_val_loss)

        # Establish canonical best model
        if best_ckpt_path.exists() and best_ckpt_path != final_best_model_path:
            shutil.copy2(best_ckpt_path, final_best_model_path)
            logger.info("Canonical best model saved to: %s", final_best_model_path)

    # =========================================================================
    # POST-TRAINING VALIDATION EVALUATION & PROFILING
    # =========================================================================
    logger.info("Loading best model for comprehensive validation evaluation...")
    eval_model = tf.keras.models.load_model(str(final_best_model_path))

    # 1. Inference Latency & Sustained Hardware Throughput
    logger.info("Profiling inference latency on dedicated accelerator...")
    latency_profile = measure_inference_latency(
        eval_model,
        input_shape=(1, 224, 224, 3),
        warmup_steps=10,
        eval_steps=50,
    )

    # 2. Validation Set Full Evaluation
    logger.info("Evaluating all %d validation samples...", n_val)
    all_preds = []
    all_trues = []
    for batch_imgs, batch_lbls in ds_val:
        preds = eval_model(batch_imgs, training=False)
        all_preds.append(preds.numpy())
        all_trues.append(batch_lbls.numpy())

    y_preds_arr = np.concatenate(all_preds, axis=0)
    y_trues_arr = np.concatenate(all_trues, axis=0)

    # 3. Multi-Class Evaluation Metrics (Accuracy, Macro P/R/F1, ROC-AUC)
    metrics = compute_classification_metrics(
        y_true=y_trues_arr,
        y_pred=y_preds_arr,
        num_classes=len(classes),
        class_names=classes,
    )

    # 4. Render and Save Confusion Matrix
    cm_plot_path = model_dir / "confusion_matrix.png"
    save_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        class_names=classes,
        output_path=cm_plot_path,
        normalize=True,
    )

    # 5. Persist comprehensive validation report
    report: Dict[str, Any] = {
        "model_architecture": args.model,
        "evaluation_dataset": "PlantVillage (group-aware validation split)",
        "validation_samples_evaluated": len(y_trues_arr),
        "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
        "training_duration_seconds": {
            "phase1_seconds": p1_time,
            "phase2_seconds": p2_time,
            "total_seconds": p1_time + p2_time,
        },
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "macro_roc_auc": metrics.get("macro_roc_auc"),
        "hardware_inference_profile": latency_profile,
        "per_class_recall": metrics["per_class_recall"],
        "per_class_roc_auc": metrics.get("per_class_roc_auc"),
        "selected_best_checkpoint": str(best_ckpt_path.name),
        "total_parameters": eval_model.count_params(),
    }

    report_path = model_dir / "val_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    cm_json_path = model_dir / "confusion_matrix.json"
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump({"confusion_matrix": metrics["confusion_matrix"].tolist()}, f)

    # Persist training configuration
    config_dict = {
        "model": args.model,
        "epochs_p1": args.epochs,
        "epochs_p2": args.fine_tune_epochs,
        "batch_size": args.batch_size,
        "learning_rate_p1": args.learning_rate,
        "learning_rate_p2": args.fine_tune_lr,
        "seed": args.seed,
        "fine_tune_layers": args.fine_tune_layers,
        "label_smoothing": args.label_smoothing,
        "patience": args.patience,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Evaluation report saved to: %s", report_path)
    logger.info("Confusion matrix saved to: %s", cm_plot_path)

    # Output Summary Table
    print("\n" + "=" * 80)
    print(f"CROPGUARD AI — PRODUCTION TRAINING COMPLETE [{args.model.upper()}]")
    print("=" * 80)
    print(f"Model Architecture  : {args.model}")
    print(f"Total Parameters    : {eval_model.count_params():,d}")
    print(f"Validation Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Macro Precision     : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall        : {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score      : {metrics['macro_f1']:.4f}")
    if metrics.get("macro_roc_auc") is not None:
        print(f"Macro ROC-AUC (OvR) : {metrics['macro_roc_auc']:.4f}")
    print(f"Inference Latency   : {latency_profile['latency_mean_ms']:.2f} ms")
    print(f"Throughput          : {latency_profile['throughput_fps']:.1f} FPS")
    print(f"Best Model File     : {final_best_model_path}")
    print(f"Evaluation Report   : {report_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    cli_args = parse_args()
    train_production_pipeline(cli_args)
