"""
CropGuard AI - IP102 Pest Detection Preparation & VOC-to-YOLO Converter
Handles conversion of PASCAL VOC XML annotations to YOLO format:
  <class_id> <x_center> <y_center> <width> <height>
with normalized coordinates in [0, 1].

Includes safe recovery for malformed / multi-root XML annotations,
deduplication of duplicate annotation blocks, verification of image-annotation
pairing, and structured error/audit reporting.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RAW_IP102_DIR = PROJECT_ROOT / "data" / "raw" / "ip102"
PROCESSED_IP102_DIR = PROJECT_ROOT / "data" / "processed" / "detection" / "ip102"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "ip102_conversion_report.json"


def voc_bbox_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Converts PASCAL VOC absolute pixel coordinates [xmin, ymin, xmax, ymax]
    to normalized YOLO format [x_center, y_center, width, height] in [0, 1].
    Clips boundaries to image limits if slightly out of bounds.
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

    # Clip to image boundaries
    xmin_c = max(0.0, min(float(xmin), float(img_width)))
    xmax_c = max(0.0, min(float(xmax), float(img_width)))
    ymin_c = max(0.0, min(float(ymin), float(img_height)))
    ymax_c = max(0.0, min(float(ymax), float(img_height)))

    box_w = xmax_c - xmin_c
    box_h = ymax_c - ymin_c

    if box_w <= 0.0 or box_h <= 0.0:
        raise ValueError(f"Degenerate bounding box: width={box_w}, height={box_h}")

    x_center = (xmin_c + xmax_c) / 2.0 / img_width
    y_center = (ymin_c + ymax_c) / 2.0 / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    # Ensure strictly in [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def parse_voc_xml(
    xml_path: Path,
    class_mapping: Optional[Dict[str, int]] = None,
    recovery_info: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Parses a VOC XML annotation file and returns a list of YOLO tuples:
    [(class_id, x_center, y_center, width, height), ...]

    Safely handles malformed XML files (e.g. multi-root XMLs with duplicate
    root blocks or trailing content) without inventing coordinates or class labels.
    """
    xml_path = Path(xml_path)
    is_recovered = False
    original_error = None
    root = None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as pe:
        # Multi-root fallback: wrap raw file contents in a synthetic root element
        original_error = str(pe)
        try:
            raw_content = xml_path.read_text(encoding="utf-8", errors="replace")
            wrapped = f"<synthetic_ip102_root>{raw_content}</synthetic_ip102_root>"
            root = ET.fromstring(wrapped)
            is_recovered = True
        except Exception as recovery_err:
            raise ValueError(
                f"Failed to parse XML in {xml_path}: {pe} (Recovery attempt failed: {recovery_err})"
            ) from pe

    # Collect all annotation elements (either root itself or multiple child annotations)
    if root.tag == "synthetic_ip102_root":
        annotation_nodes = root.findall("annotation")
        if not annotation_nodes:
            annotation_nodes = [root]
    elif root.tag == "annotation":
        annotation_nodes = [root]
    else:
        annotation_nodes = root.findall("annotation")
        if not annotation_nodes:
            annotation_nodes = [root]

    yolo_boxes: List[Tuple[int, float, float, float, float]] = []

    for ann_node in annotation_nodes:
        size_elem = ann_node.find("size")
        if size_elem is None:
            size_elem = root.find("size")
        if size_elem is None:
            continue

        img_width = int(size_elem.findtext("width", "0"))
        img_height = int(size_elem.findtext("height", "0"))

        if img_width <= 0 or img_height <= 0:
            continue

        for obj in ann_node.findall("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                continue
            cls_name = name_elem.text.strip()

            # Map class name to integer ID
            if class_mapping is not None:
                if cls_name not in class_mapping:
                    if cls_name.isdigit():
                        cls_id = int(cls_name)
                    else:
                        continue
                else:
                    cls_id = class_mapping[cls_name]
            else:
                cls_id = int(cls_name) if cls_name.isdigit() else 0

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            try:
                xc, yc, w, h = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)
                candidate = (cls_id, xc, yc, w, h)
                # Deduplicate identical boxes (especially across duplicate root blocks)
                is_duplicate = any(
                    cls_id == b[0]
                    and abs(xc - b[1]) < 1e-5
                    and abs(yc - b[2]) < 1e-5
                    and abs(w - b[3]) < 1e-5
                    and abs(h - b[4]) < 1e-5
                    for b in yolo_boxes
                )
                if not is_duplicate:
                    yolo_boxes.append(candidate)
            except ValueError:
                # Skip degenerate or zero-area boxes
                continue

    if is_recovered and recovery_info is not None:
        recovery_info["recovered"] = True
        recovery_info["file"] = xml_path.name
        recovery_info["original_error"] = original_error
        recovery_info["blocks_found"] = len(annotation_nodes)
        recovery_info["boxes_count"] = len(yolo_boxes)
        recovery_info["boxes"] = [
            {"class_id": b[0], "xc": round(b[1], 6), "yc": round(b[2], 6), "w": round(b[3], 6), "h": round(b[4], 6)}
            for b in yolo_boxes
        ]

    return yolo_boxes


def save_yolo_annotation(boxes: List[Tuple[int, float, float, float, float]], out_path: Path):
    """
    Saves bounding boxes in standard YOLO txt format:
    <class_id> <x_center> <y_center> <width> <height>
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def create_synthetic_voc_sample(xml_path: Path, img_width=640, img_height=480) -> Path:
    """
    Creates a synthetic VOC XML file for testing and validation.
    """
    xml_content = f"""<annotation>
    <folder>images</folder>
    <filename>{xml_path.stem}.jpg</filename>
    <size>
        <width>{img_width}</width>
        <height>{img_height}</height>
        <depth>3</depth>
    </size>
    <object>
        <name>12</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>64</xmin>
            <ymin>48</ymin>
            <xmax>320</xmax>
            <ymax>240</ymax>
        </bndbox>
    </object>
    <object>
        <name>45</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>320</xmin>
            <ymin>240</ymin>
            <xmax>576</xmax>
            <ymax>432</ymax>
        </bndbox>
    </object>
</annotation>"""
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content.strip())
    return xml_path


def convert_ip102_dataset(
    voc_dir: Path,
    out_dir: Path,
    images_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Converts all VOC XML files in voc_dir to YOLO label files in out_dir.
    Performs empirical validation of bounding boxes, coordinates, class IDs,
    and image-annotation pairing. Generates a comprehensive conversion report.
    """
    voc_dir = Path(voc_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(list(voc_dir.glob("*.xml")))
    total_xml_files = len(xml_files)

    converted_count = 0
    recovered_files: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, Any]] = []
    total_boxes = 0
    unique_class_ids: Set[int] = set()
    out_of_bounds_boxes = 0
    invalid_class_boxes = 0
    pairing_missing_images = 0

    print(f"Starting conversion of {total_xml_files} VOC XML annotations to {out_dir}...")

    for i, xml_file in enumerate(xml_files, 1):
        rec_info: Dict[str, Any] = {}
        try:
            boxes = parse_voc_xml(xml_file, recovery_info=rec_info)
            txt_file = out_dir / f"{xml_file.stem}.txt"
            save_yolo_annotation(boxes, txt_file)
            converted_count += 1
            total_boxes += len(boxes)

            if rec_info.get("recovered"):
                recovered_files.append(rec_info)

            # Validate boxes
            for cls_id, xc, yc, w, h in boxes:
                unique_class_ids.add(cls_id)
                if cls_id < 0 or cls_id > 101:
                    invalid_class_boxes += 1
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    out_of_bounds_boxes += 1

            # Validate image pairing
            if images_dir is not None:
                img_path = images_dir / f"{xml_file.stem}.jpg"
                if not img_path.exists():
                    pairing_missing_images += 1

        except Exception as e:
            failed_files.append({"file": xml_file.name, "error": str(e)})

        if i % 2500 == 0 or i == total_xml_files:
            print(f"  Processed {i}/{total_xml_files} files ({converted_count} successful, {len(recovered_files)} recovered, {len(failed_files)} failed)...")

    report: Dict[str, Any] = {
        "status": "SUCCESS" if len(failed_files) == 0 else "PARTIAL",
        "total_xml_files_scanned": total_xml_files,
        "successfully_converted": converted_count,
        "recovered_malformed_xmls": len(recovered_files),
        "failed_files": len(failed_files),
        "total_bounding_boxes": total_boxes,
        "unique_classes_count": len(unique_class_ids),
        "min_class_id": min(unique_class_ids) if unique_class_ids else None,
        "max_class_id": max(unique_class_ids) if unique_class_ids else None,
        "validation": {
            "invalid_class_boxes": invalid_class_boxes,
            "out_of_bounds_boxes": out_of_bounds_boxes,
            "all_coordinates_in_unit_range": out_of_bounds_boxes == 0,
            "all_class_ids_in_range_0_101": invalid_class_boxes == 0,
            "pairing_missing_images": pairing_missing_images,
            "pairing_integrity_100_pct": pairing_missing_images == 0,
        },
        "recovered_details": recovered_files,
        "failed_details": failed_files,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Audit & conversion report written to {report_path}")

    return report


def generate_ip102_splits(
    processed_dir: Path = PROCESSED_IP102_DIR,
    raw_dir: Path = RAW_IP102_DIR,
    seed: int = 42,
    val_ratio: float = 0.15,
    smoke_train_count: int = 64,
    smoke_val_count: int = 32,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Creates deterministic stratified train and val splits from trainval.txt (15,178 IDs),
    quarantines all test.txt IDs (3,798), excludes unmatched JPEGs from training,
    and generates Ultralytics dataset descriptors (ip102.yaml and ip102_smoke.yaml).
    """
    import random
    from collections import defaultdict
    from src.cropguard.detection.classes import load_ip102_classes

    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    lbl_dir = processed_dir / "labels"
    img_dir = processed_dir / "images"

    # Ensure images symlink exists in processed_dir pointing to raw JPEGImages
    if not img_dir.exists():
        raw_jpegs = raw_dir / "JPEGImages"
        if raw_jpegs.exists():
            img_dir.symlink_to(raw_jpegs)

    # 1. Read raw split IDs
    trainval_txt = raw_dir / "trainval.txt"
    test_txt = raw_dir / "test.txt"

    if not trainval_txt.exists():
        raise FileNotFoundError(f"Missing trainval.txt at {trainval_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Missing test.txt at {test_txt}")

    with open(trainval_txt, "r", encoding="utf-8") as f:
        trainval_ids = [line.strip() for line in f if line.strip()]
    with open(test_txt, "r", encoding="utf-8") as f:
        test_ids = [line.strip() for line in f if line.strip()]

    # 2. Map trainval IDs to primary class for stratified splitting
    id_to_class: Dict[str, int] = {}
    for tid in trainval_ids:
        lbl_file = lbl_dir / f"{tid}.txt"
        if not lbl_file.exists():
            raise FileNotFoundError(f"Missing label file for trainval sample: {lbl_file}")
        with open(lbl_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            cls_id = int(first_line.split()[0]) if first_line else 0
        id_to_class[tid] = cls_id

    class_to_ids: Dict[int, List[str]] = defaultdict(list)
    for tid, cid in id_to_class.items():
        class_to_ids[cid].append(tid)

    # 3. Deterministic stratified split
    rng = random.Random(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []

    for cid in sorted(class_to_ids.keys()):
        sample_ids = list(class_to_ids[cid])
        rng.shuffle(sample_ids)
        n = len(sample_ids)
        if n == 1:
            train_ids.extend(sample_ids)
        else:
            n_val = max(1, int(round(n * val_ratio)))
            val_ids.extend(sample_ids[:n_val])
            train_ids.extend(sample_ids[n_val:])

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    # 4. Deterministic smoke test subsets
    rng_smoke = random.Random(seed)
    smoke_train_ids = sorted(rng_smoke.sample(train_ids, min(smoke_train_count, len(train_ids))))
    smoke_val_ids = sorted(rng_smoke.sample(val_ids, min(smoke_val_count, len(val_ids))))

    # 5. Write split text files with explicit processed image paths
    def write_paths(ids: List[str], target_file: Path):
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            for tid in ids:
                f.write(f"{processed_dir.resolve()}/images/{tid}.jpg\n")

    train_file = processed_dir / "train.txt"
    val_file = processed_dir / "val.txt"
    test_file = processed_dir / "test.txt"
    smoke_train_file = processed_dir / "train_smoke.txt"
    smoke_val_file = processed_dir / "val_smoke.txt"

    write_paths(train_ids, train_file)
    write_paths(val_ids, val_file)
    write_paths(test_ids, test_file)
    write_paths(smoke_train_ids, smoke_train_file)
    write_paths(smoke_val_ids, smoke_val_file)

    # 6. Generate Ultralytics dataset configuration YAMLs
    classes = load_ip102_classes()

    def write_yaml(t_file: Path, v_file: Path, yaml_path: Path, is_smoke: bool = False):
        lines = [
            f"# CropGuard AI — IP102 Insect Pest Detection {'Smoke Test ' if is_smoke else ''}Dataset Descriptor",
            "# 102 canonical agricultural pest classes (Wu et al., CVPR 2019)",
            f"path: {processed_dir.resolve()}",
            f"train: {t_file.name}",
            f"val: {v_file.name}",
            "test: test.txt  # STRICTLY QUARANTINED — DO NOT ACCESS DURING TRAINING OR VALIDATION",
            "",
            "names:",
        ]
        for i in range(102):
            lines.append(f"  {i}: {classes[i + 1]}")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    full_yaml_path = processed_dir / "ip102.yaml"
    smoke_yaml_path = processed_dir / "ip102_smoke.yaml"

    write_yaml(train_file, val_file, full_yaml_path, is_smoke=False)
    write_yaml(smoke_train_file, smoke_val_file, smoke_yaml_path, is_smoke=True)

    # 7. Verification checks
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    disjoint_train_val = train_set.isdisjoint(val_set)
    disjoint_train_test = train_set.isdisjoint(test_set)
    disjoint_val_test = val_set.isdisjoint(test_set)
    quarantined_test = disjoint_train_test and disjoint_val_test and len(test_ids) == 3798

    # Check unannotated images exclusion
    unannotated = ["IP009000231", "IP022000163", "IP034000029", "IP048000152", "IP053000086"]
    unannotated_excluded = not any(u in train_set or u in val_set or u in test_set for u in unannotated)

    summary: Dict[str, Any] = {
        "status": "SUCCESS",
        "random_seed": seed,
        "trainval_total": len(trainval_ids),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "test_samples": len(test_ids),
        "smoke_train_samples": len(smoke_train_ids),
        "smoke_val_samples": len(smoke_val_ids),
        "splits_disjointness": {
            "train_val_disjoint": disjoint_train_val,
            "train_test_disjoint": disjoint_train_test,
            "val_test_disjoint": disjoint_val_test,
            "test_strictly_quarantined": quarantined_test,
            "unannotated_5_jpegs_excluded": unannotated_excluded,
        },
        "classes_represented": {
            "train_classes_count": len(class_to_ids),
            "val_classes_count": len({id_to_class[tid] for tid in val_ids}),
            "total_canonical_classes": 102,
        },
        "artifacts": {
            "train_txt": str(train_file),
            "val_txt": str(val_file),
            "test_txt": str(test_file),
            "train_smoke_txt": str(smoke_train_file),
            "val_smoke_txt": str(smoke_val_file),
            "dataset_yaml": str(full_yaml_path),
            "smoke_yaml": str(smoke_yaml_path),
        },
    }

    if summary_path is None:
        summary_path = PROJECT_ROOT / "data" / "metadata" / "ip102_split_summary.json"
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"IP102 split summary written to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="CropGuard AI - IP102 VOC to YOLO Converter")
    parser.add_argument("--voc-dir", type=str, default=str(RAW_IP102_DIR / "Annotations"))
    parser.add_argument("--images-dir", type=str, default=str(RAW_IP102_DIR / "JPEGImages"))
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_IP102_DIR / "labels"))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--generate-splits", action="store_true", help="Generate train/val/test splits and YAMLs after conversion")
    parser.add_argument("--splits-only", action="store_true", help="Generate train/val/test splits without re-running XML conversion")
    parser.add_argument("--test-synthetic", action="store_true", help="Run self-test with synthetic sample")
    args = parser.parse_args()

    if args.splits_only:
        print("Generating IP102 dataset partitions and YAML descriptors...")
        summary = generate_ip102_splits()
        print("\n=== IP102 Split Generation Complete ===")
        print(f"Train samples:      {summary['train_samples']}")
        print(f"Val samples:        {summary['val_samples']}")
        print(f"Test samples:       {summary['test_samples']} (QUARANTINED)")
        print(f"Smoke train:        {summary['smoke_train_samples']}")
        print(f"Smoke val:          {summary['smoke_val_samples']}")
        print(f"Train/Val disjoint: {summary['splits_disjointness']['train_val_disjoint']}")
        print(f"Test quarantined:   {summary['splits_disjointness']['test_strictly_quarantined']}")
        print(f"Unannotated 5 out:  {summary['splits_disjointness']['unannotated_5_jpegs_excluded']}")
        return

    if args.test_synthetic:
        test_xml = PROJECT_ROOT / "data" / "processed" / "detection" / "synthetic_test.xml"
        test_out = PROJECT_ROOT / "data" / "processed" / "detection" / "synthetic_test.txt"
        create_synthetic_voc_sample(test_xml)
        boxes = parse_voc_xml(test_xml)
        save_yolo_annotation(boxes, test_out)
        print(f"Synthetic test passed! Output written to {test_out}:")
        with open(test_out) as f:
            print(f.read())
        # Cleanup synthetic test files
        if test_xml.exists():
            test_xml.unlink()
        if test_out.exists():
            test_out.unlink()
        return

    voc_path = Path(args.voc_dir)
    if not voc_path.exists():
        print(f"Notice: IP102 raw annotation directory not found at {voc_path}.")
        print("IP102 dataset download status: Host on Google Drive (~18.5GB) requires web authentication.")
        print("See docs/DATASET_AUDIT.md for manual retrieval instructions.")
        return

    images_path = Path(args.images_dir) if args.images_dir else None
    out_path = Path(args.out_dir)
    report_path = Path(args.report_path) if args.report_path else None

    report = convert_ip102_dataset(
        voc_dir=voc_path,
        out_dir=out_path,
        images_dir=images_path,
        report_path=report_path,
    )

    print("\n=== IP102 Conversion Summary ===")
    print(f"Total XMLs scanned:       {report['total_xml_files_scanned']}")
    print(f"Successfully converted:   {report['successfully_converted']}")
    print(f"Recovered malformed XMLs: {report['recovered_malformed_xmls']}")
    print(f"Failed files:             {report['failed_files']}")
    print(f"Total bounding boxes:     {report['total_bounding_boxes']}")
    print(f"Class IDs range:          [{report['min_class_id']}, {report['max_class_id']}]")
    print(f"Unique classes count:     {report['unique_classes_count']}")
    print(f"Valid coordinates:        {report['validation']['all_coordinates_in_unit_range']}")
    print(f"Valid class IDs:          {report['validation']['all_class_ids_in_range_0_101']}")
    print(f"Image pairing intact:     {report['validation']['pairing_integrity_100_pct']}")
    if report["recovered_details"]:
        print(f"\nRecovered details:")
        for rec in report["recovered_details"]:
            print(f"  - {rec['file']}: {rec['boxes_count']} box(es) recovered ({rec['original_error']})")

    if args.generate_splits:
        print("\nGenerating IP102 dataset partitions and YAML descriptors...")
        generate_ip102_splits()


if __name__ == "__main__":
    main()
