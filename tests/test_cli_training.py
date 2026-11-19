"""
Unit tests for CropGuard AI CLI training pipeline (Milestone 08).
Verifies argument parsing, model selection, configuration validation,
dataset partition isolation, fine-tuning parameter toggles, dry-run execution,
and pure NumPy multiclass ROC-AUC metrics.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cropguard.classification.train import (
    parse_args,
    validate_dataset_partitions,
    build_model_by_name,
    execute_dry_run,
)
from src.cropguard.classification.models import set_backbone_trainable_layers
from src.cropguard.classification.preprocessing import (
    get_plantvillage_classes,
    compute_class_weights,
)
from src.cropguard.classification.evaluation import (
    compute_binary_roc_auc,
    compute_multiclass_roc_auc,
    compute_classification_metrics,
)


def test_cli_argument_parsing_defaults():
    """Verify default CLI argument values."""
    args = parse_args(["--model", "mobilenet_v2"])
    assert args.model == "mobilenet_v2"
    assert args.epochs == 10
    assert args.batch_size == 32
    assert args.learning_rate == 1e-3
    assert args.seed == 42
    assert args.fine_tune_layers == 25
    assert args.fine_tune_epochs == 5
    assert args.fine_tune_lr == 1e-5
    assert args.cache_mode == "none"
    assert args.dry_run is False


def test_cli_argument_parsing_custom_flags():
    """Verify custom CLI flag overrides."""
    args = parse_args([
        "--model", "custom_cnn",
        "--epochs", "20",
        "--batch-size", "16",
        "--learning-rate", "5e-4",
        "--dry-run",
    ])
    assert args.model == "custom_cnn"
    assert args.epochs == 20
    assert args.batch_size == 16
    assert args.learning_rate == 5e-4
    assert args.dry_run is True


def test_cli_argument_invalid_model():
    """Verify invalid model name triggers argparse rejection."""
    with pytest.raises(SystemExit):
        parse_args(["--model", "invalid_architecture_xyz"])


def test_dataset_partition_isolation_and_counts():
    """Verify full dataset partition sizes and strict disjointness."""
    train_csv, val_csv, test_csv, n_train, n_val, n_test = validate_dataset_partitions()
    assert n_train == 35221
    assert n_val == 8085
    assert n_test == 10999
    assert n_train + n_val + n_test == 54305


def test_cli_model_selection_custom_cnn():
    """Verify custom_cnn building helper returns model with no backbone."""
    model, backbone = build_model_by_name("custom_cnn", num_classes=38)
    assert model.name == "custom_foliar_cnn"
    assert backbone is None
    assert model.count_params() == 429158


def test_cli_model_selection_transfer_models():
    """Verify transfer learning models return (model, backbone) tuples."""
    mb_model, mb_backbone = build_model_by_name("mobilenet_v2", num_classes=38)
    assert mb_backbone is not None
    assert mb_model.output_shape == (None, 38)

    rn_model, rn_backbone = build_model_by_name("resnet50", num_classes=38)
    assert rn_backbone is not None
    assert rn_model.output_shape == (None, 38)


def test_fine_tuning_configuration_unfreezing():
    """Verify unfreezing top layers activates trainable weights while preserving frozen BatchNorm."""
    _, backbone = build_model_by_name("mobilenet_v2", num_classes=38)
    initial_trainable = sum(p.numpy().size for p in backbone.trainable_weights)
    assert initial_trainable == 0, "Frozen backbone should initially have 0 trainable weights"

    unfrozen_count = set_backbone_trainable_layers(backbone, num_layers_to_unfreeze=20)
    assert unfrozen_count > 0, "Unfreezing top 20 layers must introduce trainable weights"

    # Verify that BatchNormalization layers remain frozen
    for layer in backbone.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assert layer.trainable is False, "BatchNorm layers must remain frozen during fine-tuning"


def test_dry_run_execution_and_config_generation(tmp_path):
    """Verify dry-run mode completes validation and persists config.json without training."""
    args = parse_args([
        "--model", "mobilenet_v2",
        "--output-dir", str(tmp_path),
        "--dry-run",
    ])
    train_csv, val_csv, test_csv, n_train, n_val, n_test = validate_dataset_partitions()
    classes, class_to_idx, _ = get_plantvillage_classes()
    class_weights = compute_class_weights(train_csv, class_to_idx)
    model_dir = tmp_path / args.model

    summary = execute_dry_run(
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

    assert summary["dry_run"] is True
    assert summary["status"] == "VALIDATED"
    assert summary["dataset_partitions"]["test_partition_status"] == "STRICTLY_HELD_OUT_UNTOUCHED"
    assert summary["dataset_partitions"]["test_samples"] == 10999

    config_file = model_dir / "config.json"
    assert config_file.exists(), "config.json must be generated in dry-run mode"
    with open(config_file, "r", encoding="utf-8") as f:
        saved_config = json.load(f)
    assert saved_config["model_architecture"] == "mobilenet_v2"
    assert saved_config["total_parameters"] == 2600806


def test_multiclass_roc_auc_pure_numpy():
    """Verify pure NumPy multiclass ROC-AUC calculation."""
    # Synthetic ground truth for 3 classes, 6 samples
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Near-perfect predictions
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
        [0.05, 0.9, 0.05],
        [0.1, 0.1, 0.8],
    ])

    results = compute_multiclass_roc_auc(y_true, y_prob, num_classes=3)
    assert results["macro_roc_auc"] == 1.0, f"Expected perfect 1.0 ROC-AUC, got {results['macro_roc_auc']}"

    # Verify integration with compute_classification_metrics
    metrics = compute_classification_metrics(y_true, y_pred=y_prob, num_classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_roc_auc"] == 1.0
