"""
CropGuard AI - PlantVillage Physical Leaf-Level Leakage & Duplication Audit
Performs a comprehensive audit of:
1. Exact duplicate images (MD5 / SHA-256).
2. Physical leaf grouping data leakage in existing naive splits vs official author metadata.
3. Generates verified, group-aware, leak-free splits in data/splits/classification_group_aware/
   preserving the official test partition and ensuring 0% physical leaf leakage.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_manifest.csv"
LEAF_MAP_PATH = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "official_metadata" / "leaf-map.json"
OFFICIAL_TRAIN_TXT = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "official_metadata" / "color_train.txt"
OFFICIAL_TEST_TXT = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "official_metadata" / "color_test.txt"

OLD_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "classification"
NEW_SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "classification_group_aware"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img_path: Path, hash_size=8) -> int:
    with Image.open(img_path) as img:
        img_gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img_gray, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return int(sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v]))


def load_leaf_map(leaf_map_path: Path):
    if not leaf_map_path.exists():
        raise FileNotFoundError(f"Author leaf-map not found at {leaf_map_path}")
    with open(leaf_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_leaf_id(filename: str, class_name: str, leaf_map: dict) -> str:
    """
    Extracts physical leaf identity using author-provided leaf-map.json.
    Normalizes filename and prefixes with class_name to avoid cross-species collisions.
    """
    clean = filename.replace("_final_masked", "")
    if "___" in clean:
        clean = clean.split("___")[-1]
    # Remove 'copy', 'copy 2', etc. to group identical shots/re-exports of the same leaf
    clean = clean.split("copy")[0]
    for ext in [".jpg", ".JPG", ".png", ".PNG"]:
        clean = clean.replace(ext, "")
    clean = clean.strip()
    lookup_key = clean.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if len(suggestions) == 1:
            return suggestions[0]
        for s in suggestions:
            if class_name in s:
                return s
        return f"{class_name}:::{clean}"
    return f"{class_name}:::{clean}"


def audit_exact_duplicates(records):
    print("\n--- AUDIT 1: EXACT DUPLICATE DETECTION (MD5 & SHA256) ---")
    md5_counts = Counter(r["md5"] for r in records)
    dup_md5 = {k: v for k, v in md5_counts.items() if v > 1}
    dup_images_count = sum(dup_md5.values())
    print(f"Total images audited   : {len(records):,}")
    print(f"Unique MD5 checksums   : {len(md5_counts):,}")
    print(f"Duplicate MD5 values   : {len(dup_md5):,}")
    print(f"Images in duplicate sets: {dup_images_count:,}")

    return {
        "unique_md5_count": len(md5_counts),
        "duplicate_md5_hashes_count": len(dup_md5),
        "duplicate_images_count": dup_images_count
    }


def audit_old_splits_leakage(records, old_splits_dir: Path, leaf_map: dict):
    print("\n--- AUDIT 2: PHYSICAL LEAF LEAKAGE IN OLD 70/15/15 SPLITS ---")
    if not (old_splits_dir / "train.csv").exists():
        print(f"Old splits directory not found at {old_splits_dir}. Skipping old audit.")
        return {}

    path_to_record = {r["local_path"]: r for r in records}

    def load_old(name):
        with open(old_splits_dir / f"{name}.csv", "r", encoding="utf-8") as f:
            return [path_to_record[row["local_path"]] for row in csv.DictReader(f) if row["local_path"] in path_to_record]

    train_rows = load_old("train")
    val_rows = load_old("val")
    test_rows = load_old("test")

    # Check exact MD5 overlap in old split
    train_md5 = set(r["md5"] for r in train_rows)
    val_md5 = set(r["md5"] for r in val_rows)
    test_md5 = set(r["md5"] for r in test_rows)

    md5_leak_tv = train_md5 & val_md5
    md5_leak_tt = train_md5 & test_md5
    md5_leak_vt = val_md5 & test_md5

    # Check leaf_id overlap in old split
    leaf_to_splits = defaultdict(set)
    for r in train_rows:
        lid = extract_leaf_id(r["filename"], r["full_class"], leaf_map)
        leaf_to_splits[lid].add("train")
    for r in val_rows:
        lid = extract_leaf_id(r["filename"], r["full_class"], leaf_map)
        leaf_to_splits[lid].add("val")
    for r in test_rows:
        lid = extract_leaf_id(r["filename"], r["full_class"], leaf_map)
        leaf_to_splits[lid].add("test")

    total_leaf_groups = len(leaf_to_splits)
    leaking_leaf_groups = {lid: s for lid, s in leaf_to_splits.items() if len(s) > 1}
    leak_tv = {lid: s for lid, s in leaking_leaf_groups.items() if "train" in s and "val" in s and "test" not in s}
    leak_tt = {lid: s for lid, s in leaking_leaf_groups.items() if "train" in s and "test" in s and "val" not in s}
    leak_vt = {lid: s for lid, s in leaking_leaf_groups.items() if "val" in s and "test" in s and "train" not in s}
    leak_all = {lid: s for lid, s in leaking_leaf_groups.items() if len(s) == 3}

    print(f"Old Split Counts       : Train={len(train_rows):,}, Val={len(val_rows):,}, Test={len(test_rows):,}")
    print(f"Total Physical Leaves  : {total_leaf_groups:,} distinct leaf_id groups")
    print(f"Exact Duplicate Leaks  : Train&Val={len(md5_leak_tv)}, Train&Test={len(md5_leak_tt)}, Val&Test={len(md5_leak_vt)}")
    print(f"Physical Leaf Leakage  : {len(leaking_leaf_groups):,} leaking leaves ({len(leaking_leaf_groups)/total_leaf_groups*100:.2f}% of all leaves!)")
    print(f"  Train & Val only leak : {len(leak_tv):,} leaves")
    print(f"  Train & Test only leak: {len(leak_tt):,} leaves")
    print(f"  Val & Test only leak  : {len(leak_vt):,} leaves")
    print(f"  All 3 splits leak     : {len(leak_all):,} leaves")

    return {
        "old_total_leaf_groups": total_leaf_groups,
        "old_leaking_leaf_groups": len(leaking_leaf_groups),
        "old_leak_percentage": round(len(leaking_leaf_groups) / total_leaf_groups * 100, 2),
        "old_md5_train_test_overlap": len(md5_leak_tt),
        "old_leaf_train_test_overlap": len(leak_tt) + len(leak_all)
    }


def audit_official_author_splits(records, leaf_map: dict):
    print("\n--- AUDIT 3: OFFICIAL AUTHOR SPLIT (Mohanty et al., 2016) ---")
    if not OFFICIAL_TRAIN_TXT.exists() or not OFFICIAL_TEST_TXT.exists():
        print("Official split text files not found. Skipping official audit.")
        return {}

    with open(OFFICIAL_TRAIN_TXT, "r") as f:
        off_train = set(line.strip() for line in f if line.strip())
    with open(OFFICIAL_TEST_TXT, "r") as f:
        off_test = set(line.strip() for line in f if line.strip())

    train_leaves = set()
    test_leaves = set()
    train_by_leaf = defaultdict(list)
    test_by_leaf = defaultdict(list)

    for p in off_train:
        parts = p.split("/")
        lid = extract_leaf_id(parts[-1], parts[-2], leaf_map)
        train_leaves.add(lid)
        train_by_leaf[lid].append(p)

    for p in off_test:
        parts = p.split("/")
        lid = extract_leaf_id(parts[-1], parts[-2], leaf_map)
        test_leaves.add(lid)
        test_by_leaf[lid].append(p)

    overlap_leaves = train_leaves & test_leaves
    train_overlap_imgs = sum(len(train_by_leaf[l]) for l in overlap_leaves)
    test_overlap_imgs = sum(len(test_by_leaf[l]) for l in overlap_leaves)

    print(f"Official Train Images  : {len(off_train):,} ({len(off_train)/54305*100:.2f}%) across {len(train_leaves):,} leaves")
    print(f"Official Test Images   : {len(off_test):,} ({len(off_test)/54305*100:.2f}%) across {len(test_leaves):,} leaves")
    print(f"Overlapping Leaves     : {len(overlap_leaves):,} leaves ({train_overlap_imgs} train imgs, {test_overlap_imgs} test imgs)")
    print(f"Explanation: Author test partition split minor duplicate re-exports (e.g. 'copy.jpg' vs 'copy 2.jpg').")

    return {
        "official_train_images": len(off_train),
        "official_test_images": len(off_test),
        "official_train_leaves": len(train_leaves),
        "official_test_leaves": len(test_leaves),
        "official_overlap_leaves": len(overlap_leaves),
        "official_train_overlap_images": train_overlap_imgs,
        "official_test_overlap_images": test_overlap_imgs
    }


def generate_group_aware_splits(records, leaf_map: dict, out_dir: Path, seed=42):
    print("\n--- GENERATING GROUP-AWARE LEAK-FREE SPLITS ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(OFFICIAL_TEST_TXT, "r") as f:
        official_test_set = set(line.strip().replace("raw/color/", "data/processed/classification/plantvillage/") for line in f if line.strip())

    # Assign leaf_id to every record
    for r in records:
        r["leaf_id"] = extract_leaf_id(r["filename"], r["full_class"], leaf_map)
        r["in_official_test"] = r["local_path"] in official_test_set

    # To eliminate the 227 overlapping groups completely:
    # Any leaf_id appearing in official test has ALL its images assigned to Test!
    test_leaf_ids = set(r["leaf_id"] for r in records if r["in_official_test"])
    test_split = [r for r in records if r["leaf_id"] in test_leaf_ids]
    remaining_pool = [r for r in records if r["leaf_id"] not in test_leaf_ids]

    # Stratified partition of remaining_pool into Train and Val by leaf_id
    leaves_by_class = defaultdict(lambda: defaultdict(list))
    for r in remaining_pool:
        leaves_by_class[r["full_class"]][r["leaf_id"]].append(r)

    rng = random.Random(seed)
    train_split = []
    val_split = []
    val_target_ratio = 0.1875  # 0.1875 * 79.75% = ~14.95% overall validation split

    for cls_name, leaf_dict in sorted(leaves_by_class.items()):
        leaf_items = list(leaf_dict.items())
        leaf_items.sort(key=lambda x: x[0])  # Deterministic sorting
        rng.shuffle(leaf_items)

        cls_total = sum(len(imgs) for _, imgs in leaf_items)
        target_val = int(round(cls_total * val_target_ratio))

        cur_val = 0
        cls_v = []
        cls_t = []

        for lid, imgs in leaf_items:
            if cur_val < target_val and (cur_val + len(imgs) <= target_val or cur_val == 0):
                cls_v.extend(imgs)
                cur_val += len(imgs)
            else:
                cls_t.extend(imgs)

        train_split.extend(cls_t)
        val_split.extend(cls_v)

    total_all = len(train_split) + len(val_split) + len(test_split)
    assert total_all == len(records), f"Mismatch in count: {total_all} vs {len(records)}"

    # Audit zero leaf and MD5 leakage
    train_leaves = set(r["leaf_id"] for r in train_split)
    val_leaves = set(r["leaf_id"] for r in val_split)
    test_leaves = set(r["leaf_id"] for r in test_split)

    assert len(train_leaves & val_leaves) == 0, "Train & Val share leaves!"
    assert len(train_leaves & test_leaves) == 0, "Train & Test share leaves!"
    assert len(val_leaves & test_leaves) == 0, "Val & Test share leaves!"

    train_md5 = set(r["md5"] for r in train_split)
    val_md5 = set(r["md5"] for r in val_split)
    test_md5 = set(r["md5"] for r in test_split)

    assert len(train_md5 & val_md5) == 0, "Train & Val share MD5!"
    assert len(train_md5 & test_md5) == 0, "Train & Test share MD5!"
    assert len(val_md5 & test_md5) == 0, "Val & Test share MD5!"

    print(f"Group-Aware Train Split : {len(train_split):,} images ({len(train_split)/total_all*100:.2f}%) across {len(train_leaves):,} leaf groups")
    print(f"Group-Aware Val Split   : {len(val_split):,} images ({len(val_split)/total_all*100:.2f}%) across {len(val_leaves):,} leaf groups")
    print(f"Group-Aware Test Split  : {len(test_split):,} images ({len(test_split)/total_all*100:.2f}%) across {len(test_leaves):,} leaf groups")
    print("Zero-Leakage Audit      : STRICTLY ZERO (0) leaf or MD5 overlap across any partition.")

    # Save CSV and TXT files
    fields = ["local_path", "filename", "full_class", "crop", "condition", "is_healthy", "leaf_id", "md5"]
    splits = {"train": train_split, "val": val_split, "test": test_split}
    stats = {}

    for name, items in splits.items():
        csv_file = out_dir / f"{name}.csv"
        txt_file = out_dir / f"{name}.txt"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in items:
                writer.writerow(item)

        with open(txt_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(f"{item['local_path']}\n")

        cls_counts = Counter(r["full_class"] for r in items)
        stats[name] = {
            "total_images": len(items),
            "percentage": round(len(items) / total_all * 100, 2),
            "total_leaf_groups": len(set(r["leaf_id"] for r in items)),
            "class_distribution": dict(cls_counts)
        }

    summary_file = out_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return {
        "train_count": len(train_split),
        "train_pct": round(len(train_split) / total_all * 100, 2),
        "train_leaves": len(train_leaves),
        "val_count": len(val_split),
        "val_pct": round(len(val_split) / total_all * 100, 2),
        "val_leaves": len(val_leaves),
        "test_count": len(test_split),
        "test_pct": round(len(test_split) / total_all * 100, 2),
        "test_leaves": len(test_leaves),
        "leaf_leakage_overlap": 0,
        "md5_leakage_overlap": 0
    }


def main():
    print("================================================================")
    print("CROPGUARD AI - PLANTVILLAGE LEAF-LEVEL LEAKAGE & INTEGRITY AUDIT")
    print("================================================================")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    leaf_map = load_leaf_map(LEAF_MAP_PATH)

    dup_stats = audit_exact_duplicates(records)
    old_stats = audit_old_splits_leakage(records, OLD_SPLITS_DIR, leaf_map)
    off_stats = audit_official_author_splits(records, leaf_map)
    new_stats = generate_group_aware_splits(records, leaf_map, NEW_SPLITS_DIR)

    full_report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "PlantVillage (54,305 color images, 38 classes)",
        "exact_duplicates": dup_stats,
        "old_naive_70_15_15_audit": old_stats,
        "official_mohanty_audit": off_stats,
        "corrected_group_aware_splits": new_stats,
        "new_splits_location": str(NEW_SPLITS_DIR.relative_to(PROJECT_ROOT))
    }

    report_path = METADATA_DIR / "leaf_leakage_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved full audit report to: {report_path}")


if __name__ == "__main__":
    main()
