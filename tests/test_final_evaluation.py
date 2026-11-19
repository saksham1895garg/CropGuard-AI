"""
Unit tests for CropGuard AI Final Held-Out Test Evaluation CLI (Milestone 09).
Uses synthetic data and mock checkpoints. Never touches the real held-out test split.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.classification.evaluate_heldout import (
    parse_args,
    verify_test_partition_isolation,
    run_evaluation,
)


def test_cli_argument_parsing_defaults():
    """Verify default arguments of evaluation CLI."""
    args = parse_args([])
    assert args.model == "resnet50"
    assert args.batch_size == 32
    assert "models/classification/production/resnet50/best_model.keras" in args.model_path
    assert "data/splits/classification_group_aware/test.csv" in args.test_csv
    assert "reports/final_evaluation" in args.output_dir


def test_cli_argument_parsing_custom():
    """Verify custom argument overrides."""
    args = parse_args([
        "--model-path", "models/custom_ckpt.keras",
        "--test-csv", "data/custom_test.csv",
        "--output-dir", "custom_reports",
        "--model", "mobilenet_v2",
        "--batch-size", "16",
    ])
    assert args.model == "mobilenet_v2"
    assert args.batch_size == 16
    assert args.model_path == "models/custom_ckpt.keras"
    assert args.test_csv == "data/custom_test.csv"
    assert args.output_dir == "custom_reports"


def test_verify_test_partition_isolation_detects_leakage(tmp_path):
    """Ensure that artificial overlap between train and test raises an assertion error."""
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    # Shared image between train and test
    shared_img = "data/img_shared.jpg"
    img_val = "data/img_val.jpg"
    img_test_unique = "data/img_test.jpg"

    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_path", "full_class"])
        writer.writerow([shared_img, "class_0"])

    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_path", "full_class"])
        writer.writerow([img_val, "class_1"])

    with open(test_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_path", "full_class"])
        writer.writerow([shared_img, "class_0"])
        writer.writerow([img_test_unique, "class_2"])

    with pytest.raises(AssertionError, match="CRITICAL LEAKAGE"):
        verify_test_partition_isolation(test_csv)


def test_run_evaluation_synthetic_pipeline(tmp_path):
    """
    Validates end-to-end evaluation logic using a synthetic model,
    dummy JPEG images, and a temporary test CSV.
    """
    # 1. Create a lightweight synthetic classifier for 38 classes
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(38, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    mock_ckpt_path = tmp_path / "mock_model.keras"
    model.save(str(mock_ckpt_path))

    # 2. Create synthetic test images relative to PROJECT_ROOT
    # Save them in scratch/test_dummy/ to ensure local_path resolution works
    scratch_dir = PROJECT_ROOT / "scratch" / "test_eval_dummy"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []
    created_files = []
    try:
        for i in range(6):
            img_path = scratch_dir / f"test_leaf_{i}.jpg"
            img = Image.fromarray(np.uint8(np.random.randint(0, 256, (224, 224, 3))))
            img.save(img_path)
            created_files.append(img_path)
            rel_path = str(img_path.relative_to(PROJECT_ROOT))
            sample_records.append([rel_path, f"class_{i % 38}"])

        synthetic_test_csv = tmp_path / "test.csv"
        with open(synthetic_test_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["local_path", "full_class"])
            writer.writerows(sample_records)

        # 3. Execute evaluation
        output_dir = tmp_path / "reports"
        report = run_evaluation(
            model_path=mock_ckpt_path,
            test_csv=synthetic_test_csv,
            output_dir=output_dir,
            model_name="custom_cnn",
            batch_size=2,
        )

        # 4. Verify outputs
        assert report["test_samples_evaluated"] == 6
        assert "metrics" in report
        assert 0.0 <= report["metrics"]["test_accuracy"] <= 1.0
        assert 0.0 <= report["metrics"]["macro_f1"] <= 1.0

        json_report_path = output_dir / "custom_cnn_test_metrics.json"
        cm_path = output_dir / "custom_cnn_confusion_matrix.png"
        csv_path = output_dir / "custom_cnn_per_class_metrics.csv"

        assert json_report_path.exists()
        assert cm_path.exists()
        assert csv_path.exists()
        assert cm_path.stat().st_size > 0

        # Verify per-class CSV row count (38 classes + header = 39 lines)
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 39
        assert rows[0] == [
            "class_index", "class_name", "support", "tp", "fp", "fn",
            "precision", "recall", "f1_score", "roc_auc"
        ]
    finally:
        for f in created_files:
            if f.exists():
                f.unlink()
        if scratch_dir.exists():
            try:
                scratch_dir.rmdir()
            except OSError:
                pass
