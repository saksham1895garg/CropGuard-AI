"""
Unit tests for CropGuard AI U-Net segmentation training CLI pipeline (Milestone 11).
Verifies:
- Argument parsing defaults and smoke-test overrides
- Dry-run preflight checks and parameter counting
- Strict quarantine enforcement of the PlantSeg test partition
- Sample prediction visualization rendering
- End-to-end training and artifact persistence using synthetic data
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.segmentation.train import (
    parse_args,
    train_segmentation_pipeline,
    save_prediction_visualizations,
)
from src.cropguard.segmentation.models import build_unet


def test_parse_args_defaults():
    """Verify default CLI argument values."""
    args = parse_args([])
    assert args.epochs == 20
    assert args.batch_size == 8
    assert args.learning_rate == 1e-4
    assert args.bce_weight == 0.5
    assert args.dice_weight == 0.5
    assert args.patience_early == 5
    assert args.patience_lr == 3
    assert args.base_filters == 32
    assert args.depth == 4
    assert args.dry_run is False
    assert args.smoke_test is False
    assert args.output_dir == "models/segmentation/production/unet"


def test_parse_args_smoke_test_overrides():
    """Verify --smoke-test flag automatically applies rapid verification defaults."""
    args = parse_args(["--smoke-test"])
    assert args.smoke_test is True
    assert args.epochs == 2
    assert args.subset_train == 32
    assert args.subset_val == 16
    assert args.output_dir == "models/segmentation/smoke_test/unet"


def test_parse_args_custom_overrides():
    """Verify custom CLI flag overrides."""
    args = parse_args([
        "--epochs", "15",
        "--batch-size", "16",
        "--learning-rate", "5e-4",
        "--base-filters", "16",
        "--depth", "3",
        "--output-dir", "custom/output/dir",
        "--dry-run",
    ])
    assert args.epochs == 15
    assert args.batch_size == 16
    assert args.learning_rate == 5e-4
    assert args.base_filters == 16
    assert args.depth == 3
    assert args.output_dir == "custom/output/dir"
    assert args.dry_run is True


def test_dry_run_execution():
    """Verify dry-run preflight checks execute without fitting the model."""
    args = parse_args(["--dry-run"])
    result = train_segmentation_pipeline(args)

    assert result["status"] == "DRY_RUN_PASSED"
    assert result["training_samples_available"] == 7916
    assert result["validation_samples_available"] == 1247
    assert result["test_partition_status"] == "STRICTLY_HELD_OUT_UNTOUCHED"
    assert result["total_parameters"] == 7771873
    assert result["trainable_parameters"] == 7765985
    assert result["model_architecture"] == "U-Net"


def test_test_partition_quarantine_enforcement():
    """Verify pipeline rejects any attempt to feed test.csv into training or validation."""
    test_csv = PROJECT_ROOT / "data" / "splits" / "segmentation" / "test.csv"

    # Attempt to pass test.csv as train_csv
    args_train_violation = parse_args(["--train-csv", str(test_csv), "--dry-run"])
    with pytest.raises(AssertionError, match="CRITICAL"):
        train_segmentation_pipeline(args_train_violation)

    # Attempt to pass test.csv as val_csv
    args_val_violation = parse_args(["--val-csv", str(test_csv), "--dry-run"])
    with pytest.raises(AssertionError, match="CRITICAL"):
        train_segmentation_pipeline(args_val_violation)


def test_save_prediction_visualizations(tmp_path):
    """Verify sample prediction visualization renders non-empty PNG images."""
    imgs = np.random.rand(4, 256, 256, 3).astype(np.float32)
    gts = (np.random.rand(4, 256, 256, 1) > 0.8).astype(np.float32)
    preds = np.random.rand(4, 256, 256, 1).astype(np.float32)

    saved = save_prediction_visualizations(
        images=imgs,
        ground_truths=gts,
        predictions=preds,
        output_dir=tmp_path / "sample_predictions",
        num_samples=3,
        threshold=0.5,
    )

    assert len(saved) == 3
    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 1000  # Non-trivial image file


def test_synthetic_end_to_end_training_and_artifacts(tmp_path):
    """
    Verifies end-to-end training loop, metrics computation, and artifact persistence
    using synthetic lightweight images on a compact U-Net.
    """
    # 1. Create synthetic image and mask files
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    train_rows = ["image_filename,mask_filename,local_image_path,local_mask_path,dataset_origin\n"]
    val_rows = ["image_filename,mask_filename,local_image_path,local_mask_path,dataset_origin\n"]

    for i in range(4):
        im_arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        ma_arr = (np.random.rand(64, 64) > 0.7).astype(np.uint8) * 255

        im_path = img_dir / f"img_{i}.jpg"
        ma_path = mask_dir / f"mask_{i}.png"
        Image.fromarray(im_arr).save(im_path)
        Image.fromarray(ma_arr).save(ma_path)

        row = f"img_{i}.jpg,mask_{i}.png,{im_path},{ma_path},synthetic\n"
        if i < 2:
            train_rows.append(row)
        else:
            val_rows.append(row)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train_csv.write_text("".join(train_rows))
    val_csv.write_text("".join(val_rows))

    # 2. Build mock zip path (dummy file since local paths are provided)
    dummy_zip = tmp_path / "dummy.zip"
    dummy_zip.touch()

    out_dir = tmp_path / "out"

    # 3. Configure lightweight run
    args = parse_args([
        "--train-csv", str(train_csv),
        "--val-csv", str(val_csv),
        "--zip-path", str(dummy_zip),
        "--output-dir", str(out_dir),
        "--epochs", "1",
        "--batch-size", "2",
        "--base-filters", "8",
        "--depth", "2",
    ])

    results = train_segmentation_pipeline(args)

    # 4. Assert artifacts exist
    assert (out_dir / "best_model.keras").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "validation_metrics.json").exists()
    assert (out_dir / "training_history.csv").exists()
    assert (out_dir / "sample_predictions").is_dir()
    assert len(list((out_dir / "sample_predictions").glob("*.png"))) >= 1

    # 5. Assert metric structures
    with open(out_dir / "validation_metrics.json", "r") as f:
        metrics = json.load(f)
    assert "dice_coefficient" in metrics
    assert "iou_score" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "specificity" in metrics
    assert "mean_severity_delta_pct" in metrics
    assert metrics["test_partition_status"] == "STRICTLY_HELD_OUT_UNTOUCHED"
