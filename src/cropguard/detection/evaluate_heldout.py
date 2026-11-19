"""
CropGuard AI - Final Held-Out Test Evaluation CLI for YOLO26n Pest Detection.
Evaluates the frozen production YOLO26n checkpoint strictly against
the official held-out IP102 test partition (3,798 images).
Generates structured metrics (mAP@50, mAP@50-95, precision, recall, latency)
and per-class performance tables without modifying the model.
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.detection.classes import format_pest_name
from src.cropguard.detection.inference import predict_pests, render_pest_detections

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cropguard.detection.evaluate_heldout")

DEFAULT_WEIGHTS = (
    PROJECT_ROOT / "models" / "pest_detection" / "production" / "yolo26n" / "weights" / "best.pt"
)
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102" / "ip102.yaml"
DEFAULT_TEST_TXT = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102" / "test.txt"
DEFAULT_TRAIN_TXT = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102" / "train.txt"
DEFAULT_VAL_TXT = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102" / "val.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"


def verify_test_isolation(
    train_txt: Path, val_txt: Path, test_txt: Path
) -> Dict[str, Any]:
    """Verifies that the test set has strict zero overlap with train and val partitions."""
    if not test_txt.exists():
        raise FileNotFoundError(f"Test split file not found: {test_txt}")

    with open(test_txt, "r", encoding="utf-8") as f:
        test_samples = [line.strip() for line in f if line.strip()]

    train_samples = []
    if train_txt.exists():
        with open(train_txt, "r", encoding="utf-8") as f:
            train_samples = [line.strip() for line in f if line.strip()]

    val_samples = []
    if val_txt.exists():
        with open(val_txt, "r", encoding="utf-8") as f:
            val_samples = [line.strip() for line in f if line.strip()]

    test_set = set(test_samples)
    train_set = set(train_samples)
    val_set = set(val_samples)

    train_test_overlap = len(test_set.intersection(train_set))
    val_test_overlap = len(test_set.intersection(val_set))
    train_val_overlap = len(train_set.intersection(val_set))

    if train_test_overlap > 0 or val_test_overlap > 0:
        raise ValueError(
            f"CRITICAL: Test partition data leakage detected! "
            f"Train/Test overlap={train_test_overlap}, Val/Test overlap={val_test_overlap}"
        )

    return {
        "test_samples_count": len(test_samples),
        "train_samples_count": len(train_samples),
        "val_samples_count": len(val_samples),
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "train_val_overlap": train_val_overlap,
        "disjointness_verified": True,
    }


def save_test_visualizations(
    model: YOLO,
    test_list_path: Path,
    output_dir: Path,
    num_samples: int = 6,
    conf_threshold: float = 0.25,
) -> List[Path]:
    """Saves visual prediction diagnostic overlays on representative held-out test images."""
    sample_dir = output_dir / "yolo26n_sample_predictions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if not test_list_path.exists():
        return saved_paths

    with open(test_list_path, "r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()][:num_samples]

    for p in image_paths:
        if not p.exists():
            continue
        try:
            with Image.open(p) as img:
                res = predict_pests(model, img, conf_threshold=conf_threshold)
                rendered = render_pest_detections(img, res["detections"])
                out_file = sample_dir / f"test_pred_{p.stem}.png"
                rendered.save(out_file)
                saved_paths.append(out_file)
        except Exception as e:
            logger.warning(f"Failed to generate visualization for {p.name}: {e}")

    return saved_paths


def evaluate_heldout_yolo(
    weights_path: Path,
    data_yaml_path: Path,
    test_txt_path: Path,
    output_dir: Path,
    device: Optional[str] = None,
    workers: int = 2,
    num_sample_images: int = 6,
) -> Dict[str, Any]:
    """Runs final evaluation of frozen YOLO26n on official held-out IP102 test set."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML descriptor not found: {data_yaml_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify Partition Isolation
    isolation = verify_test_isolation(
        train_txt=DEFAULT_TRAIN_TXT,
        val_txt=DEFAULT_VAL_TXT,
        test_txt=test_txt_path,
    )
    logger.info(
        f"Verified test split isolation: {isolation['test_samples_count']} images, 0% leakage."
    )

    # 2. Device configuration
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading frozen checkpoint: {weights_path} on device={dev}...")
    model = YOLO(str(weights_path))

    param_count = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") and model.model is not None else 0
    num_classes = len(model.names) if hasattr(model, "names") and isinstance(model.names, dict) else 0

    logger.info(f"Model architecture: YOLO26n ({param_count:,} parameters, {num_classes} classes)")
    logger.info(f"Starting test evaluation on {isolation['test_samples_count']} held-out images...")

    t0 = time.time()
    results = model.val(
        data=str(data_yaml_path),
        split="test",
        workers=workers,
        device=dev,
        plots=False,
        save=False,
        verbose=True,
    )
    total_eval_duration = round(time.time() - t0, 2)
    logger.info(f"Evaluation completed in {total_eval_duration}s.")

    # 3. Extract Global Metrics
    precision = round(float(results.box.mp), 4)
    recall = round(float(results.box.mr), 4)
    map50 = round(float(results.box.map50), 4)
    map50_95 = round(float(results.box.map), 4)
    fitness = round(float(results.fitness), 4)

    speed_info = {
        "preprocess_ms": round(float(results.speed.get("preprocess", 0.0)), 2),
        "inference_ms": round(float(results.speed.get("inference", 0.0)), 2),
        "postprocess_ms": round(float(results.speed.get("postprocess", 0.0)), 2),
    }
    speed_info["total_latency_ms"] = round(
        speed_info["preprocess_ms"] + speed_info["inference_ms"] + speed_info["postprocess_ms"], 2
    )
    speed_info["throughput_fps"] = round(
        1000.0 / speed_info["total_latency_ms"], 1
    ) if speed_info["total_latency_ms"] > 0 else 0.0

    # 4. Extract Per-Class Metrics
    per_class_records: List[Dict[str, Any]] = []
    ap_class_indices = getattr(results.box, "ap_class_index", [])
    all_ap = getattr(results.box, "all_ap", np.array([]))
    maps = getattr(results.box, "maps", np.array([]))

    for idx, c in enumerate(ap_class_indices):
        raw_name = model.names.get(int(c), f"class_{c}")
        display_name = format_pest_name(raw_name)
        try:
            p_val, r_val, ap50_val, ap50_95_val = results.box.class_result(idx)
        except Exception:
            ap50_val = float(all_ap[idx, 0]) if len(all_ap) > idx else 0.0
            ap50_95_val = float(maps[int(c)]) if len(maps) > int(c) else 0.0
            p_val = float(results.box.p[idx]) if hasattr(results.box, "p") and len(results.box.p) > idx else 0.0
            r_val = float(results.box.r[idx]) if hasattr(results.box, "r") and len(results.box.r) > idx else 0.0

        per_class_records.append({
            "class_id": int(c),
            "raw_name": raw_name,
            "display_name": display_name,
            "precision": round(float(p_val), 4),
            "recall": round(float(r_val), 4),
            "ap50": round(float(ap50_val), 4),
            "ap50_95": round(float(ap50_95_val), 4),
        })

    # Save per-class CSV
    per_class_csv_path = output_dir / "yolo26n_test_per_class_metrics.csv"
    if per_class_records:
        with open(per_class_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["class_id", "raw_name", "display_name", "precision", "recall", "ap50", "ap50_95"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_class_records)
        logger.info(f"Saved per-class test metrics to {per_class_csv_path}")

    # 5. Diagnostic Prediction Overlays
    sample_preds = save_test_visualizations(
        model=model,
        test_list_path=test_txt_path,
        output_dir=output_dir,
        num_samples=num_sample_images,
        conf_threshold=0.25,
    )

    # 6. Structured JSON Report
    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "YOLO26n",
        "checkpoint_path": str(weights_path),
        "total_parameters": param_count,
        "test_dataset": "IP102 (official held-out test partition)",
        "test_list_path": str(test_txt_path),
        "test_samples_evaluated": isolation["test_samples_count"],
        "distinct_classes_evaluated": len(per_class_records),
        "total_canonical_classes": num_classes,
        "isolation_status": {
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "test_strictly_held_out": True,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
            "fitness": fitness,
        },
        "hardware_inference_profile": {
            "device": dev,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_eval_duration_seconds": total_eval_duration,
            "latency_ms": speed_info,
        },
        "sample_visualizations_count": len(sample_preds),
        "per_class_metrics_file": per_class_csv_path.name if per_class_records else None,
    }

    report_json_path = output_dir / "yolo26n_test_metrics.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved held-out test evaluation report to {report_json_path}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CropGuard AI — YOLO26n Held-Out Test Evaluation"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to trained YOLO weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
        help=f"Path to dataset YAML (default: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--test-txt",
        type=str,
        default=str(DEFAULT_TEST_TXT),
        help=f"Path to test split image list (default: {DEFAULT_TEST_TXT})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for evaluation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader workers (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to evaluate on ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of diagnostic sample prediction overlays to render",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("CropGuard AI — YOLO26n Final Held-Out Test Evaluation")
    print("=" * 72)

    report = evaluate_heldout_yolo(
        weights_path=Path(args.weights),
        data_yaml_path=Path(args.data),
        test_txt_path=Path(args.test_txt),
        output_dir=Path(args.output_dir),
        device=args.device,
        workers=args.workers,
        num_sample_images=args.num_samples,
    )

    print("\n" + "=" * 72)
    print("FINAL HELD-OUT TEST EVALUATION RESULTS — YOLO26n")
    print("=" * 72)
    print(f"Test Samples:       {report['test_samples_evaluated']:,} images")
    print(f"Precision:          {report['metrics']['precision'] * 100:.2f}%")
    print(f"Recall:             {report['metrics']['recall'] * 100:.2f}%")
    print(f"mAP@50:             {report['metrics']['map50'] * 100:.2f}%")
    print(f"mAP@50-95:          {report['metrics']['map50_95'] * 100:.2f}%")
    print(f"Inference Latency:  {report['hardware_inference_profile']['latency_ms']['inference_ms']} ms/image")
    print(f"Total Latency:      {report['hardware_inference_profile']['latency_ms']['total_latency_ms']} ms/image")
    print(f"Throughput:         {report['hardware_inference_profile']['latency_ms']['throughput_fps']} FPS")
    print(f"Report Saved:       {args.output_dir}/yolo26n_test_metrics.json")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
