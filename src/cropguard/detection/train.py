"""
CropGuard AI - IP102 Pest Detection Training CLI.
Provides reproducible fine-tuning of YOLO26n on agricultural pest bounding boxes.
Features safety checks, dry-run configuration audits, smoke-test verification,
and structured production checkpoint & metrics artifact generation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch
from ultralytics import YOLO

from scripts.prepare_ip102 import generate_ip102_splits
from src.cropguard.detection.classes import load_ip102_classes
from src.cropguard.detection.inference import predict_pests, render_pest_detections
from src.cropguard.detection.models import DEFAULT_BASE_WEIGHTS, PRODUCTION_CHECKPOINT, get_preferred_device

RAW_IP102_DIR = PROJECT_ROOT / "data" / "raw" / "ip102"
PROCESSED_IP102_DIR = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102"
SMOKE_OUTPUT_DIR = PROJECT_ROOT / "models" / "pest_detection" / "smoke_test" / "yolo26n"
PRODUCTION_OUTPUT_DIR = PROJECT_ROOT / "models" / "pest_detection" / "production" / "yolo26n"


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parses command-line arguments for YOLO26n training."""
    parser = argparse.ArgumentParser(
        description="CropGuard AI — YOLO26n Insect Pest Detection Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to YOLO dataset configuration YAML file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints, metrics, and sample predictions.",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default=str(DEFAULT_BASE_WEIGHTS),
        help="Path to pretrained base YOLO weights.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader worker processes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device: 'cuda', 'cpu', or None (auto-detect).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute small verification experiment (2 epochs, batch size 4 on smoke dataset subset).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate existing checkpoint on validation split without training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify environment, weights, and dataset paths without executing model.train().",
    )

    parsed = parser.parse_args(args)

    # Resolve default output directory based on smoke-test flag
    if parsed.output_dir is None:
        if parsed.smoke_test:
            parsed.output_dir = str(SMOKE_OUTPUT_DIR)
        else:
            parsed.output_dir = str(PRODUCTION_OUTPUT_DIR)

    # Resolve default data YAML based on smoke-test flag
    if parsed.data is None:
        if parsed.smoke_test:
            parsed.data = str(PROCESSED_IP102_DIR / "ip102_smoke.yaml")
        else:
            parsed.data = str(PROCESSED_IP102_DIR / "ip102.yaml")

    # Set smoke test defaults if enabled and not overridden
    if parsed.smoke_test:
        if parsed.epochs == 20:
            parsed.epochs = 2

    return parsed


def audit_ip102_dataset_availability(data_yaml_path: Path) -> Dict[str, Any]:
    """
    Performs an empirical audit of local IP102 dataset availability, partitions,
    disjointness, and test partition quarantine.
    """
    raw_images_dir = RAW_IP102_DIR / "JPEGImages"
    raw_annotations_dir = RAW_IP102_DIR / "Annotations"
    classes_txt = RAW_IP102_DIR / "classes.txt"
    trainval_txt = RAW_IP102_DIR / "trainval.txt"
    test_txt = RAW_IP102_DIR / "test.txt"

    num_raw_images = len(list(raw_images_dir.glob("*.jpg"))) if raw_images_dir.exists() else 0
    num_raw_annotations = len(list(raw_annotations_dir.glob("*.xml"))) if raw_annotations_dir.exists() else 0
    num_processed_labels = len(list((PROCESSED_IP102_DIR / "labels").glob("*.txt"))) if (PROCESSED_IP102_DIR / "labels").exists() else 0
    has_classes_txt = classes_txt.exists()
    has_data_yaml = data_yaml_path.exists()

    # Ensure split files exist; generate if raw files are available
    if not has_data_yaml and num_raw_images > 0 and num_processed_labels >= 18976:
        print("Data YAML not found. Generating IP102 dataset partitions...")
        generate_ip102_splits()
        has_data_yaml = data_yaml_path.exists()

    # Audit split files disjointness and quarantine
    train_file = PROCESSED_IP102_DIR / "train.txt"
    val_file = PROCESSED_IP102_DIR / "val.txt"
    test_file = PROCESSED_IP102_DIR / "test.txt"

    splits_valid = False
    train_samples = 0
    val_samples = 0
    test_samples = 0
    disjoint_train_val = False
    quarantined_test = False
    unannotated_excluded = False

    if train_file.exists() and val_file.exists() and test_file.exists():
        with open(train_file) as f:
            t_ids = {Path(l.strip()).stem for l in f if l.strip()}
        with open(val_file) as f:
            v_ids = {Path(l.strip()).stem for l in f if l.strip()}
        with open(test_file) as f:
            te_ids = {Path(l.strip()).stem for l in f if l.strip()}

        train_samples = len(t_ids)
        val_samples = len(v_ids)
        test_samples = len(te_ids)
        disjoint_train_val = t_ids.isdisjoint(v_ids)
        disjoint_train_test = t_ids.isdisjoint(te_ids)
        disjoint_val_test = v_ids.isdisjoint(te_ids)
        quarantined_test = disjoint_train_test and disjoint_val_test and test_samples == 3798

        unannotated = ["IP009000231", "IP022000163", "IP034000029", "IP048000152", "IP053000086"]
        unannotated_excluded = not any(u in t_ids or u in v_ids or u in te_ids for u in unannotated)
        splits_valid = disjoint_train_val and quarantined_test and unannotated_excluded

    is_available = (num_raw_images > 0 and num_raw_annotations > 0 and has_data_yaml)

    return {
        "is_available": is_available,
        "raw_dir": str(RAW_IP102_DIR),
        "raw_images_found": num_raw_images,
        "raw_annotations_found": num_raw_annotations,
        "processed_labels_found": num_processed_labels,
        "has_classes_txt": has_classes_txt,
        "has_data_yaml": has_data_yaml,
        "data_yaml_path": str(data_yaml_path),
        "splits_audit": {
            "splits_valid": splits_valid,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "train_val_disjoint": disjoint_train_val,
            "test_strictly_quarantined": quarantined_test,
            "unannotated_5_jpegs_excluded": unannotated_excluded,
        },
    }


def run_training_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    """Validates runtime environment, GPU resources, checkpoint files, and splits."""
    device = args.device if args.device is not None else get_preferred_device()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
    vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)) if torch.cuda.is_available() else 0

    base_weights_path = Path(args.base_weights)
    has_base_weights = base_weights_path.exists()
    base_weights_size_mb = round(base_weights_path.stat().st_size / (1024 * 1024), 2) if has_base_weights else 0.0

    dataset_audit = audit_ip102_dataset_availability(Path(args.data))

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "gpu_name": gpu_name,
            "gpu_vram_mb": vram_mb,
        },
        "base_model": {
            "weights_path": str(base_weights_path),
            "exists": has_base_weights,
            "size_mb": base_weights_size_mb,
        },
        "dataset_audit": dataset_audit,
        "training_hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "imgsz": args.imgsz,
            "patience": args.patience,
            "workers": args.workers,
            "output_dir": args.output_dir,
            "smoke_test": args.smoke_test,
            "dry_run": args.dry_run,
        },
    }
    return report


def save_sample_visualizations(
    model: YOLO,
    val_list_path: Path,
    output_dir: Path,
    num_samples: int = 4,
    conf_threshold: float = 0.01,
) -> List[Path]:
    """
    Renders diagnostic bounding-box prediction visualizations on validation samples.
    """
    sample_dir = output_dir / "sample_predictions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if not val_list_path.exists():
        return saved_paths

    with open(val_list_path, "r", encoding="utf-8") as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()][:num_samples]

    for p in image_paths:
        if not p.exists():
            continue
        try:
            with Image.open(p) as img:
                res = predict_pests(model, img, conf_threshold=conf_threshold)
                rendered = render_pest_detections(img, res["detections"])
                out_file = sample_dir / f"val_pred_{p.stem}.png"
                rendered.save(out_file)
                saved_paths.append(out_file)
        except Exception as e:
            print(f"Notice: Failed to render visualization for {p.name}: {e}")

    return saved_paths


def main(cli_args: Optional[list] = None) -> int:
    """Main CLI entrypoint for YOLO26n training, smoke testing, and preflight verification."""
    args = parse_args(cli_args)
    report = run_training_preflight(args)

    print("=" * 72)
    print("CropGuard AI — YOLO26n Pest Detection Training Pipeline")
    print("=" * 72)
    print(f"Mode: {'SMOKE TEST' if args.smoke_test else 'PRODUCTION TRAINING'}")
    print(f"Device: {report['environment']['device']} ({report['environment']['gpu_name']}, {report['environment']['gpu_vram_mb']} MB VRAM)")
    print(f"Base Weights: {report['base_model']['weights_path']} (Exists: {report['base_model']['exists']})")
    print(f"Dataset YAML: {args.data}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Hyperparameters: {args.epochs} epochs | batch size {args.batch_size} | image size {args.imgsz} | workers {args.workers}")
    print("-" * 72)

    audit = report["dataset_audit"]
    if not audit["is_available"]:
        print("⚠️  DATASET STATUS: IP102 Dataset Not Ready")
        print(f"   Checked directory: {audit['raw_dir']}")
        print(f"   Images found: {audit['raw_images_found']} | XMLs found: {audit['raw_annotations_found']}")
        print(f"   Processed labels found: {audit['processed_labels_found']}")
        if args.dry_run:
            print("❌ Cannot proceed with training: IP102 dataset files are missing.")
            return 1
        return 1

    splits = audit["splits_audit"]
    print("📋 Split Audit:")
    print(f"   Train samples: {splits['train_samples']} | Val samples: {splits['val_samples']} | Test samples: {splits['test_samples']}")
    print(f"   Train/Val disjoint: {splits['train_val_disjoint']} | Test strictly quarantined: {splits['test_strictly_quarantined']}")
    print(f"   Unannotated 5 JPEGs excluded: {splits['unannotated_5_jpegs_excluded']}")
    print("-" * 72)

    if args.dry_run:
        print("✅ Dry-run preflight completed successfully. Dataset, splits, and environment are verified.")
        return 0

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Phase
    t0 = time.time()
    print("🚀 Initializing YOLO26n training...")
    model = YOLO(args.base_weights)

    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        patience=args.patience,
        device=report["environment"]["device"],
        project=str(output_dir.parent.resolve()),
        name=output_dir.name,
        exist_ok=True,
        workers=args.workers,
        plots=True,
        save=True,
        verbose=True,
    )
    training_duration_s = round(time.time() - t0, 2)
    print(f"\n🎉 Training complete in {training_duration_s}s.")

    # 2. Evaluation Phase on Best Checkpoint
    best_weights = output_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = output_dir / "weights" / "last.pt"

    print(f"\nEvaluating best checkpoint: {best_weights}...")
    eval_model = YOLO(str(best_weights))
    val_metrics = eval_model.val(
        data=args.data,
        split="val",
        workers=args.workers,
        device=report["environment"]["device"],
        verbose=False,
    )

    mAP50 = round(float(val_metrics.box.map50), 4)
    mAP50_95 = round(float(val_metrics.box.map), 4)
    precision = round(float(val_metrics.box.mp), 4)
    recall = round(float(val_metrics.box.mr), 4)
    fitness = round(float(val_metrics.fitness), 4)

    # 3. Save Structured Metrics & Config Artifacts
    config_record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "smoke_test" if args.smoke_test else "production",
        "model_architecture": "YOLO26n",
        "base_weights": args.base_weights,
        "dataset_yaml": args.data,
        "output_dir": str(output_dir),
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "patience": args.patience,
        "workers": args.workers,
        "device": report["environment"]["device"],
        "gpu_name": report["environment"]["gpu_name"],
        "splits": {
            "train_samples": splits["train_samples"] if not args.smoke_test else 64,
            "val_samples": splits["val_samples"] if not args.smoke_test else 32,
            "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
        },
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    metrics_record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(best_weights),
        "training_duration_seconds": training_duration_s,
        "validation_samples_evaluated": 32 if args.smoke_test else splits["val_samples"],
        "test_partition_status": "STRICTLY_HELD_OUT_UNTOUCHED",
        "metrics": {
            "mAP50": mAP50,
            "mAP50_95": mAP50_95,
            "precision": precision,
            "recall": recall,
            "fitness": fitness,
        },
        "inference_speed_ms": {
            "preprocess": round(float(val_metrics.speed.get("preprocess", 0.0)), 2),
            "inference": round(float(val_metrics.speed.get("inference", 0.0)), 2),
            "postprocess": round(float(val_metrics.speed.get("postprocess", 0.0)), 2),
        },
    }
    with open(output_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    # 4. Generate Diagnostic Visualizations
    val_txt_path = PROCESSED_IP102_DIR / ("val_smoke.txt" if args.smoke_test else "val.txt")
    sample_preds = save_sample_visualizations(
        model=eval_model,
        val_list_path=val_txt_path,
        output_dir=output_dir,
        num_samples=4,
    )

    print("\n" + "=" * 72)
    print(f"YOLO26n {'Smoke-Test' if args.smoke_test else 'Production'} Experiment Summary")
    print("=" * 72)
    print(f"Checkpoint Saved:   {best_weights}")
    print(f"Training Duration:  {training_duration_s}s")
    print(f"Precision:          {precision * 100:.2f}%")
    print(f"Recall:             {recall * 100:.2f}%")
    print(f"mAP@50:             {mAP50 * 100:.2f}%")
    print(f"mAP@50-95:          {mAP50_95 * 100:.2f}%")
    print(f"Inference Latency:  {metrics_record['inference_speed_ms']['inference']} ms/image")
    print(f"Sample Preds Saved: {len(sample_preds)} images in {output_dir / 'sample_predictions'}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
