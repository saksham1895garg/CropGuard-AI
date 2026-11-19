"""
CropGuard AI - PlantVillage Preparation Pipeline
Extracts, validates, indexes, and profiles the PlantVillage dataset.
Generates:
- data/metadata/dataset_manifest.csv
- data/metadata/class_distribution.csv
- data/metadata/data_quality_report.json
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "plantvillage" / "data.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "classification" / "plantvillage"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def parse_class_name(folder_name: str):
    """
    Parses PlantVillage folder name into (crop, condition, is_healthy).
    Example: 'Tomato___Bacterial_spot' -> ('Tomato', 'Bacterial_spot', False)
             'Pepper,_bell___healthy' -> ('Pepper_bell', 'healthy', True)
             'Corn_(maize)___Common_rust_' -> ('Corn_maize', 'Common_rust', False)
    """
    clean_name = folder_name.strip()
    if "___" in clean_name:
        crop_part, cond_part = clean_name.split("___", 1)
    else:
        crop_part = clean_name
        cond_part = "unknown"

    crop = crop_part.replace(",", "").replace(" ", "_").replace("(", "").replace(")", "")
    cond = cond_part.strip("_").replace(" ", "_")
    is_healthy = "healthy" in cond.lower()
    return crop, cond, is_healthy


def extract_plantvillage(zip_path: Path, dest_dir: Path, crops_filter=None, limit_per_class=None):
    """
    Extracts color images from data.zip to processed directory.
    Filters by crop family if specified.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_records = []

    if not zip_path.exists():
        raise FileNotFoundError(f"Source archive not found: {zip_path}")

    print(f"Opening archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # We only extract color images from raw/color/
        all_members = [
            m for m in zf.infolist()
            if m.filename.startswith("raw/color/") and not m.is_dir()
        ]
        print(f"Found {len(all_members)} color image candidates in archive.")

        class_counters = Counter()
        for idx, member in enumerate(all_members):
            parts = member.filename.split("/")
            if len(parts) < 3:
                continue
            folder_name = parts[2]
            filename = parts[-1]

            crop, cond, is_healthy = parse_class_name(folder_name)
            if crops_filter and crop.lower() not in [c.lower() for c in crops_filter]:
                continue

            if limit_per_class and class_counters[folder_name] >= limit_per_class:
                continue

            class_counters[folder_name] += 1
            target_class_dir = dest_dir / folder_name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_class_dir / filename

            # Extract if not exists or size mismatch
            if not target_file.exists() or target_file.stat().st_size != member.file_size:
                with zf.open(member) as src_f, open(target_file, "wb") as dst_f:
                    dst_f.write(src_f.read())

            extracted_records.append({
                "zip_path": member.filename,
                "local_path": str(target_file.relative_to(PROJECT_ROOT)),
                "filename": filename,
                "full_class": folder_name,
                "crop": crop,
                "condition": cond,
                "is_healthy": is_healthy,
                "file_size_bytes": member.file_size
            })

            if (idx + 1) % 10000 == 0 or (idx + 1) == len(all_members):
                print(f"Processed {idx + 1}/{len(all_members)} images...")

    print(f"Extraction complete. Total images extracted/indexed: {len(extracted_records)}")
    return extracted_records


def audit_and_profile_images(records):
    """
    Validates image headers, reads dimensions and channels, detects corrupt files.
    Calculates MD5 hash for deduplication/integrity tracking.
    """
    print("Auditing image integrity and extracting spatial metadata...")
    valid_records = []
    corrupted_count = 0
    channel_counts = Counter()
    dimension_counts = Counter()

    for idx, r in enumerate(records):
        abs_path = PROJECT_ROOT / r["local_path"]
        try:
            with open(abs_path, "rb") as f:
                content = f.read()
                file_md5 = hashlib.md5(content).hexdigest()

            with Image.open(abs_path) as img:
                img.verify()

            # Reopen to read mode and dimensions (verify closes or invalidates image in some PIL versions)
            with Image.open(abs_path) as img:
                w, h = img.size
                mode = img.mode

            channels = 3 if mode == "RGB" else (1 if mode in ("L", "1") else 4)
            channel_counts[mode] += 1
            dimension_counts[f"{w}x{h}"] += 1

            r["width"] = w
            r["height"] = h
            r["channels"] = channels
            r["mode"] = mode
            r["md5"] = file_md5
            valid_records.append(r)

        except Exception as e:
            print(f"Warning: Corrupt or unreadable image {abs_path}: {e}")
            corrupted_count += 1

        if (idx + 1) % 10000 == 0 or (idx + 1) == len(records):
            print(f"Audited {idx + 1}/{len(records)} images...")

    return valid_records, corrupted_count, channel_counts, dimension_counts


def write_manifests(records, metadata_dir: Path, corrupted_count, channel_counts, dimension_counts):
    """
    Writes dataset_manifest.csv, class_distribution.csv, and data_quality_report.json.
    """
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = metadata_dir / "dataset_manifest.csv"
    dist_csv = metadata_dir / "class_distribution.csv"
    report_json = metadata_dir / "data_quality_report.json"

    # 1. Dataset Manifest CSV
    fields = [
        "local_path", "filename", "full_class", "crop",
        "condition", "is_healthy", "width", "height", "channels",
        "mode", "file_size_bytes", "md5"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Written manifest to: {manifest_csv} ({len(records)} entries)")

    # 2. Class Distribution CSV
    class_counts = Counter(r["full_class"] for r in records)
    total_imgs = len(records)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[0])

    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "class_name", "crop", "condition", "count", "percentage"])
        for idx, (cls_name, cnt) in enumerate(sorted_classes):
            crop, cond, _ = parse_class_name(cls_name)
            pct = (cnt / total_imgs) * 100.0 if total_imgs > 0 else 0.0
            writer.writerow([idx, cls_name, crop, cond, cnt, f"{pct:.2f}%"])
    print(f"Written class distribution to: {dist_csv} ({len(sorted_classes)} classes)")

    # 3. Data Quality Report JSON
    min_samples = min(class_counts.values()) if class_counts else 0
    max_samples = max(class_counts.values()) if class_counts else 0
    imbalance_ratio = (max_samples / min_samples) if min_samples > 0 else 0.0

    quality_report = {
        "dataset_name": "PlantVillage (Mohanty et al., 2016)",
        "source": "Hugging Face / mohanty/PlantVillage",
        "license": "CC-BY-SA-3.0",
        "total_images_extracted": total_imgs,
        "corrupted_images_detected": corrupted_count,
        "total_classes": len(sorted_classes),
        "min_class_samples": min_samples,
        "max_class_samples": max_samples,
        "imbalance_ratio_max_to_min": round(imbalance_ratio, 2),
        "color_modes": dict(channel_counts),
        "resolutions": dict(dimension_counts),
        "crop_summary": dict(Counter(r["crop"] for r in records))
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    print(f"Written data quality report to: {report_json}")


def main():
    parser = argparse.ArgumentParser(description="CropGuard AI - PlantVillage Preparation")
    parser.add_argument("--crops", nargs="*", default=None,
                        help="Filter by crop name(s), e.g. Tomato Potato Pepper_bell. Default: all crops.")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Optional limit on images per class for fast testing.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip zip extraction if files already present.")
    args = parser.parse_args()

    records = extract_plantvillage(
        RAW_ZIP, PROCESSED_DIR, crops_filter=args.crops, limit_per_class=args.limit_per_class
    )
    valid_records, corrupt_cnt, ch_counts, dim_counts = audit_and_profile_images(records)
    write_manifests(valid_records, METADATA_DIR, corrupt_cnt, ch_counts, dim_counts)
    print("\nPlantVillage processing completed successfully!")


if __name__ == "__main__":
    main()
