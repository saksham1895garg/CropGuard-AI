"""
CropGuard AI - Data Pipeline Test Suite
Tests dataset manifest integrity, stratified leak-free split guarantees,
physical leaf-grouping disjointness, exact duplicate detection,
PASCAL VOC to YOLO coordinate transformation mathematics,
PlantSeg image-mask pairing verification, tf.data memory-safe caching,
and class distribution consistency.
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def manifest_path():
    p = PROJECT_ROOT / "data" / "metadata" / "dataset_manifest.csv"
    if not p.exists():
        pytest.skip("PlantVillage dataset manifest not yet generated.")
    return p


@pytest.fixture
def legacy_splits_dir():
    d = PROJECT_ROOT / "data" / "splits" / "classification"
    if not (d / "train.csv").exists():
        pytest.skip("Legacy classification splits not found.")
    return d


@pytest.fixture
def group_aware_splits_dir():
    d = PROJECT_ROOT / "data" / "splits" / "classification_group_aware"
    if not (d / "train.csv").exists():
        pytest.skip("Group-aware classification splits not yet generated.")
    return d


def test_manifest_structure_and_dimensions(manifest_path):
    """Verifies manifest headers, record count, and spatial uniformities."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    assert len(records) == 54305, f"Expected 54,305 records, got {len(records)}"
    required_cols = {"local_path", "filename", "full_class", "crop", "condition", "is_healthy", "width", "height", "channels", "md5"}
    assert required_cols.issubset(set(reader.fieldnames)), f"Missing required columns in manifest: {required_cols - set(reader.fieldnames)}"

    # Check first 500 samples for dimension uniformity
    for r in records[:500]:
        assert int(r["width"]) == 256
        assert int(r["height"]) == 256
        assert int(r["channels"]) in (3, 4)


def test_leak_free_stratified_splits(legacy_splits_dir):
    """Verifies that legacy splits have zero file-path overlap."""
    def load_split(name):
        with open(legacy_splits_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    train_rows = load_split("train")
    val_rows = load_split("val")
    test_rows = load_split("test")

    train_paths = set(r["local_path"] for r in train_rows)
    val_paths = set(r["local_path"] for r in val_rows)
    test_paths = set(r["local_path"] for r in test_rows)

    # 1. Zero file-path leakage
    assert len(train_paths & val_paths) == 0, "Train and Val splits share overlapping filepaths!"
    assert len(train_paths & test_paths) == 0, "Train and Test splits share overlapping filepaths!"
    assert len(val_paths & test_paths) == 0, "Val and Test splits share overlapping filepaths!"

    total = len(train_paths) + len(val_paths) + len(test_paths)
    assert total == 54305


def test_group_aware_split_disjointness(group_aware_splits_dir):
    """Verifies that group-aware splits have strictly ZERO physical leaf-id and ZERO MD5 overlap."""
    def load_split(name):
        with open(group_aware_splits_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    train_rows = load_split("train")
    val_rows = load_split("val")
    test_rows = load_split("test")

    total = len(train_rows) + len(val_rows) + len(test_rows)
    assert total == 54305, f"Expected 54,305 total samples, got {total}"

    # Verify zero leaf_id overlap
    train_leaves = set(r["leaf_id"] for r in train_rows)
    val_leaves = set(r["leaf_id"] for r in val_rows)
    test_leaves = set(r["leaf_id"] for r in test_rows)

    assert len(train_leaves & val_leaves) == 0, f"Train and Val share {len(train_leaves & val_leaves)} physical leaves!"
    assert len(train_leaves & test_leaves) == 0, f"Train and Test share {len(train_leaves & test_leaves)} physical leaves!"
    assert len(val_leaves & test_leaves) == 0, f"Val and Test share {len(val_leaves & test_leaves)} physical leaves!"

    # Verify zero MD5 checksum overlap
    train_md5 = set(r["md5"] for r in train_rows)
    val_md5 = set(r["md5"] for r in val_rows)
    test_md5 = set(r["md5"] for r in test_rows)

    assert len(train_md5 & val_md5) == 0, "Train and Val share exact duplicate MD5 hashes!"
    assert len(train_md5 & test_md5) == 0, "Train and Test share exact duplicate MD5 hashes!"
    assert len(val_md5 & test_md5) == 0, "Val and Test share exact duplicate MD5 hashes!"


def test_duplicate_leakage_audit_report():
    """Verifies the audit report for duplicate detection and leakage quantification."""
    report_file = PROJECT_ROOT / "data" / "metadata" / "leaf_leakage_audit_report.json"
    if not report_file.exists():
        pytest.skip("Leaf leakage audit report not generated.")

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    dup_stats = report["exact_duplicates"]
    assert dup_stats["duplicate_md5_hashes_count"] == 21, "Expected 21 duplicate MD5 hashes."
    assert dup_stats["duplicate_images_count"] == 42, "Expected 42 duplicate images."

    old_stats = report["old_naive_70_15_15_audit"]
    assert old_stats["old_leaking_leaf_groups"] > 6000, "Expected >6,000 leaking leaf groups in naive split."

    new_stats = report["corrected_group_aware_splits"]
    assert new_stats["leaf_leakage_overlap"] == 0, "Corrected splits must have 0 leaf leakage."
    assert new_stats["md5_leakage_overlap"] == 0, "Corrected splits must have 0 MD5 leakage."


def test_split_reproducibility():
    """Verifies that group-aware split logic produces identical partitions across repeated runs with seed=42."""
    from scripts.audit_plantvillage_leakage import load_leaf_map, generate_group_aware_splits, LEAF_MAP_PATH, MANIFEST_PATH

    leaf_map = load_leaf_map(LEAF_MAP_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    temp_out_1 = Path(tempfile.mkdtemp(prefix="split_rep_1_"))
    temp_out_2 = Path(tempfile.mkdtemp(prefix="split_rep_2_"))

    try:
        stats_1 = generate_group_aware_splits(records, leaf_map, temp_out_1, seed=42)
        stats_2 = generate_group_aware_splits(records, leaf_map, temp_out_2, seed=42)

        assert stats_1["train_count"] == stats_2["train_count"]
        assert stats_1["val_count"] == stats_2["val_count"]
        assert stats_1["test_count"] == stats_2["test_count"]

        # Check exact line-for-line file equality
        with open(temp_out_1 / "train.txt") as f1, open(temp_out_2 / "train.txt") as f2:
            assert f1.read() == f2.read(), "Train partition is non-deterministic!"
        with open(temp_out_1 / "val.txt") as f1, open(temp_out_2 / "val.txt") as f2:
            assert f1.read() == f2.read(), "Val partition is non-deterministic!"
        with open(temp_out_1 / "test.txt") as f1, open(temp_out_2 / "test.txt") as f2:
            assert f1.read() == f2.read(), "Test partition is non-deterministic!"
    finally:
        shutil.rmtree(temp_out_1, ignore_errors=True)
        shutil.rmtree(temp_out_2, ignore_errors=True)


def test_plantseg_official_splits_disjoint():
    """Verifies that PlantSeg segmentation splits are 100% mutually exclusive and match official counts."""
    seg_splits_dir = PROJECT_ROOT / "data" / "splits" / "segmentation"
    if not (seg_splits_dir / "train.csv").exists():
        pytest.skip("PlantSeg segmentation splits not found.")

    def load_seg(name):
        with open(seg_splits_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    train_rows = load_seg("train")
    val_rows = load_seg("val")
    test_rows = load_seg("test")

    assert len(train_rows) == 7916, f"Expected 7,916 train rows, got {len(train_rows)}"
    assert len(val_rows) == 1247, f"Expected 1,247 val rows, got {len(val_rows)}"
    assert len(test_rows) == 2295, f"Expected 2,295 test rows, got {len(test_rows)}"

    total = len(train_rows) + len(val_rows) + len(test_rows)
    assert total == 11458

    train_imgs = set(r["image_name"] for r in train_rows)
    val_imgs = set(r["image_name"] for r in val_rows)
    test_imgs = set(r["image_name"] for r in test_rows)

    assert len(train_imgs & val_imgs) == 0, "PlantSeg Train & Val share images!"
    assert len(train_imgs & test_imgs) == 0, "PlantSeg Train & Test share images!"
    assert len(val_imgs & test_imgs) == 0, "PlantSeg Val & Test share images!"


def test_cache_configuration(group_aware_splits_dir):
    """Tests tf.data pipeline cache configuration: default None, optional disk, and invalid mode error."""
    from src.cropguard.data.loader import build_classification_dataset

    train_csv = group_aware_splits_dir / "train.csv"

    # 1. Default safe mode: cache_mode=None (no caching)
    ds_none = build_classification_dataset(train_csv, batch_size=4, limit=16, cache_mode=None)
    batch_img, batch_lbl = next(iter(ds_none))
    assert batch_img.shape == (4, 224, 224, 3)

    # 2. Disk caching mode
    temp_cache_dir = Path(tempfile.mkdtemp(prefix="tf_cache_test_"))
    try:
        ds_disk = build_classification_dataset(
            train_csv, batch_size=4, limit=16, cache_mode="disk", cache_dir=temp_cache_dir
        )
        # Iterate once to trigger caching
        for _ in ds_disk:
            pass
        cache_files = list(temp_cache_dir.glob("*"))
        assert len(cache_files) > 0, "Disk cache file was not created!"
    finally:
        shutil.rmtree(temp_cache_dir, ignore_errors=True)

    # 3. Invalid cache mode raises ValueError
    with pytest.raises(ValueError, match="Unsupported cache_mode"):
        build_classification_dataset(train_csv, batch_size=4, limit=16, cache_mode="invalid_cache")


def test_dataset_class_distribution():
    """Verifies that class_distribution.csv contains all 38 classes and accurate statistics."""
    dist_file = PROJECT_ROOT / "data" / "metadata" / "class_distribution.csv"
    if not dist_file.exists():
        pytest.skip("class_distribution.csv not found.")

    with open(dist_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 38, f"Expected 38 classes, found {len(reader)}"

    total_samples = sum(int(r["count"]) for r in reader)
    assert total_samples == 54305, f"Expected 54,305 total samples, got {total_samples}"

    counts = [int(r["count"]) for r in reader]
    assert max(counts) == 5507, "Expected max class sample count 5,507 (Citrus Greening)"
    assert min(counts) == 152, "Expected min class sample count 152 (Potato Healthy)"


def test_voc_to_yolo_conversion_math():
    """Verifies mathematical correctness of VOC XML to YOLO bounding box conversion."""
    from scripts.prepare_ip102 import voc_bbox_to_yolo

    xc, yc, w, h = voc_bbox_to_yolo(64, 48, 320, 240, 640, 480)
    assert pytest.approx(xc, abs=1e-5) == 0.3
    assert pytest.approx(yc, abs=1e-5) == 0.3
    assert pytest.approx(w, abs=1e-5) == 0.4
    assert pytest.approx(h, abs=1e-5) == 0.4

    # Boundary clipping test
    xc_clip, yc_clip, w_clip, h_clip = voc_bbox_to_yolo(-10, -5, 700, 500, 640, 480)
    assert 0.0 <= xc_clip <= 1.0
    assert 0.0 <= yc_clip <= 1.0
    assert pytest.approx(w_clip, abs=1e-5) == 1.0
    assert pytest.approx(h_clip, abs=1e-5) == 1.0

    # Degenerate box test (zero or negative area)
    with pytest.raises(ValueError, match="Degenerate bounding box"):
        voc_bbox_to_yolo(100, 100, 100, 200, 640, 480)


def test_segmentation_pairing_and_metrics():
    """Tests image-mask pair verification and lesion coverage calculation."""
    from scripts.prepare_plantseg import create_synthetic_plantseg_sample, verify_image_mask_pair

    temp_dir = Path(tempfile.mkdtemp(prefix="cropguard_seg_test_"))
    try:
        img_f, mask_f = create_synthetic_plantseg_sample(temp_dir, sample_id="test_leaf", width=200, height=200)
        w, h, lesions, coverage = verify_image_mask_pair(img_f, mask_f)

        assert w == 200
        assert h == 200
        assert lesions > 0
        assert 0.0 < coverage < 100.0

        # Test mismatched dimensions
        mismatch_mask = temp_dir / "mismatch_mask.png"
        Image.new("L", (100, 100), 0).save(mismatch_mask)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            verify_image_mask_pair(img_f, mismatch_mask)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
