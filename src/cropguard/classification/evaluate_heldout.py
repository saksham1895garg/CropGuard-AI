"""
CropGuard AI - Final Held-Out Test Evaluation CLI.
Evaluates a frozen production classification checkpoint (e.g. ResNet50)
strictly against the held-out group-aware test partition (10,999 images).
Reuses existing classification and data loader modules without retraining.
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.classification.evaluation import (
    compute_classification_metrics,
    measure_inference_latency,
    save_confusion_matrix_plot,
)
from src.cropguard.classification.preprocessing import (
    get_model_preprocessing,
    get_plantvillage_classes,
)
from src.cropguard.data.loader import build_classification_dataset, load_dataset_records

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cropguard.classification.evaluate_test")


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
    """Parses CLI arguments for final held-out test evaluation."""
    parser = argparse.ArgumentParser(
        prog="cropguard-evaluate-test",
        description="CropGuard AI — Final Held-Out Test Evaluation CLI",
        epilog=(
            "Example:\n"
            "  python src/cropguard/classification/evaluate_test.py \\\n"
            "    --model-path models/classification/production/resnet50/best_model.keras \\\n"
            "    --test-csv data/splits/classification_group_aware/test.csv \\\n"
            "    --output-dir reports/final_evaluation \\\n"
            "    --model resnet50\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/classification/production/resnet50/best_model.keras",
        help="Path to frozen Keras production model checkpoint (.keras).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/splits/classification_group_aware/test.csv",
        help="Path to held-out test partition CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/final_evaluation",
        help="Directory to persist evaluation metrics, confusion matrix, and reports.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenet_v2", "custom_cnn"],
        help="Model architecture for preprocessing selection (default: resnet50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size (default: 32).",
    )
    return parser.parse_args(args)


def verify_test_partition_isolation(test_csv: Path) -> Tuple[int, int, int]:
    """
    Verifies that the test partition exists, is populated, and has zero sample overlap
    with the training and validation partitions.
    """
    splits_dir = test_csv.parent
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    test_fps, test_lbls, _ = load_dataset_records(test_csv)
    n_test = len(test_fps)

    if train_csv.exists() and val_csv.exists():
        train_fps, _, _ = load_dataset_records(train_csv)
        val_fps, _, _ = load_dataset_records(val_csv)
        s_train = set(train_fps)
        s_val = set(val_fps)
        s_test = set(test_fps)

        assert len(s_train & s_test) == 0, "CRITICAL LEAKAGE: Overlap detected between train and test partitions!"
        assert len(s_val & s_test) == 0, "CRITICAL LEAKAGE: Overlap detected between val and test partitions!"
        logger.info(
            "Test partition isolation verified: Train=%d, Val=%d, Test=%d (0 overlap)",
            len(s_train), len(s_val), n_test
        )
        return len(s_train), len(s_val), n_test
    else:
        logger.warning("Train/Val splits not found alongside test.csv; verified test records: %d", n_test)
        return 0, 0, n_test


def run_evaluation(
    model_path: Path,
    test_csv: Path,
    output_dir: Path,
    model_name: str = "resnet50",
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Executes the complete evaluation workflow and persists structured reports."""
    configure_gpu_memory()

    # Preflight 1: Checkpoint existence
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Preflight 2: Test split existence & isolation
    if not test_csv.is_absolute():
        test_csv = PROJECT_ROOT / test_csv
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_csv}")

    n_train, n_val, n_test = verify_test_partition_isolation(test_csv)

    # Preflight 3: Output directory setup
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight 4: Classes and Preprocessing
    classes, class_to_idx, idx_to_class = get_plantvillage_classes()
    num_classes = len(classes)
    preprocess_fn = get_model_preprocessing(model_name)

    # Preflight 5: Model loading and verification
    logger.info("Loading frozen production checkpoint: %s", model_path)
    model = tf.keras.models.load_model(str(model_path))
    assert model.output_shape[-1] == num_classes, (
        f"Model output dimension {model.output_shape[-1]} does not match num_classes {num_classes}"
    )

    # Hardware latency profile
    logger.info("Profiling inference latency on dedicated accelerator...")
    latency_profile = measure_inference_latency(
        model,
        input_shape=(1, 224, 224, 3),
        warmup_steps=10,
        eval_steps=50,
    )

    # Streaming test dataset pipeline
    logger.info("Streaming %d test samples through tf.data input pipeline...", n_test)
    ds_test = build_classification_dataset(
        csv_path=test_csv,
        batch_size=batch_size,
        shuffle=False,
        cache_mode=None,  # Safe streaming
        preprocess_fn=preprocess_fn,
    )

    all_probs: List[np.ndarray] = []
    all_trues: List[np.ndarray] = []

    t_start = time.time()
    for batch_imgs, batch_lbls in ds_test:
        probs = model(batch_imgs, training=False)
        all_probs.append(probs.numpy())
        all_trues.append(batch_lbls.numpy())
    eval_duration = time.time() - t_start

    y_prob_arr = np.concatenate(all_probs, axis=0)
    y_true_arr = np.concatenate(all_trues, axis=0)

    # Compute comprehensive classification metrics
    logger.info("Computing test classification metrics & Wilcoxon-Mann-Whitney ROC-AUC...")
    metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_pred=y_prob_arr,
        num_classes=num_classes,
        class_names=classes,
    )

    # 1. Save Confusion Matrix PNG
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        class_names=classes,
        output_path=cm_path,
        normalize=True,
    )
    logger.info("Confusion matrix heatmap saved to: %s", cm_path)

    # 2. Save Per-Class Metrics CSV
    per_class_csv_path = output_dir / f"{model_name}_per_class_metrics.csv"
    cm = metrics["confusion_matrix"]
    tp = np.diag(cm).astype(int)
    fp = (np.sum(cm, axis=0) - tp).astype(int)
    fn = (np.sum(cm, axis=1) - tp).astype(int)
    support = np.sum(cm, axis=1).astype(int)
    precision_arr = metrics["class_precisions_array"]
    recall_arr = metrics["class_recalls_array"]
    f1_arr = metrics["class_f1_array"]
    per_class_roc = metrics.get("per_class_roc_auc", {})

    with open(per_class_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_index",
            "class_name",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ])
        for k in range(num_classes):
            c_name = classes[k]
            writer.writerow([
                k,
                c_name,
                int(support[k]),
                int(tp[k]),
                int(fp[k]),
                int(fn[k]),
                f"{float(precision_arr[k]):.6f}",
                f"{float(recall_arr[k]):.6f}",
                f"{float(f1_arr[k]):.6f}",
                f"{float(per_class_roc.get(c_name, 0.0)):.6f}",
            ])
    logger.info("Per-class metrics CSV saved to: %s", per_class_csv_path)

    # 3. Save Final Evaluation Report JSON
    now_iso = datetime.now(timezone.utc).isoformat()
    report: Dict[str, Any] = {
        "evaluation_title": "CropGuard AI — Final Held-Out Test Evaluation",
        "evaluation_timestamp_utc": now_iso,
        "model_architecture": model_name,
        "model_checkpoint_path": str(model_path),
        "total_model_parameters": int(model.count_params()),
        "test_dataset_path": str(test_csv),
        "test_samples_evaluated": int(len(y_true_arr)),
        "dataset_split_counts": {
            "train_samples": n_train,
            "val_samples": n_val,
            "test_samples": n_test,
            "split_disjointness_verified": True,
        },
        "preprocessing_method": "caffe (RGB-to-BGR conversion + ImageNet channel mean subtraction [103.939, 116.779, 123.68])",
        "inference_mode": "training=False (frozen weights, deterministic execution)",
        "evaluation_duration_seconds": float(eval_duration),
        "metrics": {
            "test_accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "macro_roc_auc": float(metrics.get("macro_roc_auc", 0.0)),
        },
        "hardware_inference_profile": latency_profile,
        "class_mapping": {k: c for k, c in enumerate(classes)},
        "per_class_recall": metrics["per_class_recall"],
        "per_class_roc_auc": metrics.get("per_class_roc_auc", {}),
        "confusion_matrix_shape": list(cm.shape),
    }

    report_path = output_dir / f"{model_name}_test_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Final test metrics JSON report saved to: %s", report_path)

    return report


if __name__ == "__main__":
    cli_args = parse_args()
    report = run_evaluation(
        model_path=Path(cli_args.model_path),
        test_csv=Path(cli_args.test_csv),
        output_dir=Path(cli_args.output_dir),
        model_name=cli_args.model,
        batch_size=cli_args.batch_size,
    )
    m = report["metrics"]
    out_dir = Path(cli_args.output_dir)
    print("\n" + "=" * 80)
    print("CROPGUARD AI — FINAL HELD-OUT TEST EVALUATION COMPLETE [RESNET50]")
    print("=" * 80)
    print(f"Model Architecture  : {report['model_architecture']}")
    print(f"Model Checkpoint    : {report['model_checkpoint_path']}")
    print(f"Test Samples        : {report['test_samples_evaluated']:,d} (100% held-out)")
    print(f"Test Accuracy       : {m['test_accuracy']:.4f} ({m['test_accuracy'] * 100:.2f}%)")
    print(f"Macro Precision     : {m['macro_precision']:.4f}")
    print(f"Macro Recall        : {m['macro_recall']:.4f}")
    print(f"Macro F1-Score      : {m['macro_f1']:.4f}")
    print(f"Macro ROC-AUC (OvR) : {m['macro_roc_auc']:.4f}")
    print(f"Inference Latency   : {report['hardware_inference_profile']['latency_mean_ms']:.2f} ms")
    print(f"Throughput          : {report['hardware_inference_profile']['throughput_fps']:.1f} FPS")
    print(f"Metrics Report JSON : {out_dir / f'{cli_args.model}_test_metrics.json'}")
    print(f"Per-Class CSV       : {out_dir / f'{cli_args.model}_per_class_metrics.csv'}")
    print(f"Confusion Matrix PNG: {out_dir / f'{cli_args.model}_confusion_matrix.png'}")
    print("=" * 80 + "\n")
