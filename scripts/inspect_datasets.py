"""
CropGuard AI - Dataset Inspection CLI Utility
Audits repository data status: raw archives, processed directories, manifests,
split distributions, and checks for data integrity / leaks.
"""

import csv
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def format_bytes(b: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024.0:
            return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"


def inspect_raw_archives():
    print("=" * 60)
    print("RAW ARCHIVES AUDIT")
    print("=" * 60)

    raw_dir = DATA_DIR / "raw"
    sources = [
        ("PlantVillage (Classification)", raw_dir / "plantvillage" / "data.zip"),
        ("PlantSeg (Foliar Segmentation)", raw_dir / "plantseg" / "Plantseg.zip"),
        ("IP102 (Pest Detection)", raw_dir / "ip102")
    ]

    for name, path in sources:
        if path.is_file():
            size_str = format_bytes(path.stat().st_size)
            print(f"[FOUND] {name:35}: File present ({size_str})")
        elif path.is_dir():
            files = list(path.glob("**/*"))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"[FOUND] {name:35}: Directory ({file_count} files, {format_bytes(total_size)})")
        else:
            print(f"[ABSENT] {name:34}: Not present on disk at {path.relative_to(PROJECT_ROOT)}")


def inspect_processed_data():
    print("\n" + "=" * 60)
    print("PROCESSED DATA AUDIT")
    print("=" * 60)

    pv_processed = DATA_DIR / "processed" / "classification" / "plantvillage"
    if pv_processed.exists():
        classes = [d for d in pv_processed.iterdir() if d.is_dir()]
        total_imgs = sum(len(list(c.glob("*.*"))) for c in classes)
        print(f"PlantVillage Classification: {len(classes)} classes, {total_imgs} images")
    else:
        print("PlantVillage Classification: Not yet extracted into processed/classification/plantvillage")

    seg_processed = DATA_DIR / "processed" / "segmentation" / "plantseg"
    if seg_processed.exists():
        img_cnt = len(list((seg_processed / "images").glob("*.*"))) if (seg_processed / "images").exists() else 0
        mask_cnt = len(list((seg_processed / "masks").glob("*.*"))) if (seg_processed / "masks").exists() else 0
        print(f"PlantSeg Segmentation: {img_cnt} images, {mask_cnt} masks")
    else:
        print("PlantSeg Segmentation: Not yet processed")

    det_processed = DATA_DIR / "processed" / "detection" / "ip102"
    if det_processed.exists():
        labels = len(list((det_processed / "labels").glob("*.txt"))) if (det_processed / "labels").exists() else 0
        print(f"IP102 Detection: {labels} YOLO label files")
    else:
        print("IP102 Detection: Not yet processed")


def inspect_metadata_and_splits():
    print("\n" + "=" * 60)
    print("METADATA & SPLIT INTEGRITY AUDIT")
    print("=" * 60)

    manifest = DATA_DIR / "metadata" / "dataset_manifest.csv"
    if manifest.exists():
        with open(manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Dataset Manifest: {len(rows)} indexed records in {manifest.name}")
    else:
        print(f"Dataset Manifest: {manifest.name} not found")

    splits_dir = DATA_DIR / "splits" / "classification"
    train_f = splits_dir / "train.csv"
    val_f = splits_dir / "val.csv"
    test_f = splits_dir / "test.csv"

    if train_f.exists() and val_f.exists() and test_f.exists():
        def read_paths(p):
            with open(p, "r", encoding="utf-8") as f:
                return set(row["local_path"] for row in csv.DictReader(f))

        train_s = read_paths(train_f)
        val_s = read_paths(val_f)
        test_s = read_paths(test_f)

        print(f"Classification Splits: Train={len(train_s)}, Val={len(val_s)}, Test={len(test_s)}")
        overlap = (train_s & val_s) | (train_s & test_s) | (val_s & test_s)
        if overlap:
            print(f"[FAIL] CRITICAL: Found {len(overlap)} overlapping samples across splits!")
        else:
            print("[PASS] ZERO LEAKAGE: All splits are completely disjoint.")
    else:
        print("Classification Splits: Incomplete or missing")


def main():
    inspect_raw_archives()
    inspect_processed_data()
    inspect_metadata_and_splits()
    print("\nInspection audit finished.")


if __name__ == "__main__":
    main()
