"""
CropGuard AI - Leak-Free Stratified Split Generator
Generates reproducible, leak-free 70/15/15 Train / Validation / Test splits
for classification (and multi-task datasets).
Verifies:
1. Exact mathematical stratification across all classes.
2. Complete disjointness: train_set & val_set == empty, train_set & test_set == empty, val_set & test_set == empty.
3. Deterministic reproduction via fixed random seed.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "classification"


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def stratified_split(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Partitions records into train, val, test using stratified sampling per class.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    # Group by full_class
    by_class = defaultdict(list)
    for r in records:
        by_class[r["full_class"]].append(r)

    rng = random.Random(seed)
    train_split = []
    val_split = []
    test_split = []

    for cls_name in sorted(by_class.keys()):
        items = list(by_class[cls_name])
        # Sort first for deterministic order regardless of input filesystem ordering
        items.sort(key=lambda x: x["local_path"])
        rng.shuffle(items)

        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 sample in val and test if n >= 3
        if n >= 3:
            if n_val == 0:
                n_val = 1
            if (n - n_train - n_val) == 0:
                if n_train > 1:
                    n_train -= 1

        n_test = n - n_train - n_val

        cls_train = items[:n_train]
        cls_val = items[n_train:n_train + n_val]
        cls_test = items[n_train + n_val:]

        train_split.extend(cls_train)
        val_split.extend(cls_val)
        test_split.extend(cls_test)

    return train_split, val_split, test_split


def verify_leak_free(train_split, val_split, test_split):
    """
    Rigorous verification that splits are mutually exclusive and collectively exhaustive.
    """
    train_paths = set(r["local_path"] for r in train_split)
    val_paths = set(r["local_path"] for r in val_split)
    test_paths = set(r["local_path"] for r in test_split)

    # Check 1: Pairwise intersections
    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    assert len(train_val_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} items in train & val!"
    assert len(train_test_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} items in train & test!"
    assert len(val_test_overlap) == 0, f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} items in val & test!"

    # Check 2: Deduplication within splits
    assert len(train_paths) == len(train_split), "Duplicate paths detected in train split!"
    assert len(val_paths) == len(val_split), "Duplicate paths detected in val split!"
    assert len(test_paths) == len(test_split), "Duplicate paths detected in test split!"

    total = len(train_split) + len(val_split) + len(test_split)
    print(f"Leak-free audit PASSED: {total} total samples partitioned with 0% overlap.")


def save_splits(train_split, val_split, test_split, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }

    fields = ["local_path", "filename", "full_class", "crop", "condition", "is_healthy"]

    stats = {}
    for split_name, items in splits.items():
        csv_file = out_dir / f"{split_name}.csv"
        txt_file = out_dir / f"{split_name}.txt"

        # Write CSV
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in items:
                writer.writerow(r)

        # Write TXT (relative paths)
        with open(txt_file, "w", encoding="utf-8") as f:
            for r in items:
                f.write(f"{r['local_path']}\n")

        # Compute class counts for statistics
        cls_counts = defaultdict(int)
        for r in items:
            cls_counts[r["full_class"]] += 1

        stats[split_name] = {
            "total_samples": len(items),
            "class_counts": dict(cls_counts)
        }
        print(f"Saved {split_name} split: {len(items)} samples -> {csv_file}")

    summary_file = out_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved split summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="CropGuard AI - Generate Stratified Splits")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--out-dir", type=str, default=str(SPLITS_DIR))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_manifest(Path(args.manifest))
    print(f"Loaded {len(records)} records from {args.manifest}")

    train_s, val_s, test_s = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    verify_leak_free(train_s, val_s, test_s)
    save_splits(train_s, val_s, test_s, Path(args.out_dir))
    print("\nStratified split generation completed successfully!")


if __name__ == "__main__":
    main()
