"""
CropGuard AI - Pest Detection Unit & Regression Tests.
Verifies VOC-to-YOLO conversion, IP102 taxonomy, YOLO26n model loading,
inference parsing, empty-detection handling, visualization rendering, and CLI safety.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_ip102 import (
    create_synthetic_voc_sample,
    parse_voc_xml,
    voc_bbox_to_yolo,
)
from src.cropguard.detection.classes import (
    IP102_CLASSES_CANONICAL,
    format_pest_name,
    get_ip102_class_list,
    load_ip102_classes,
)
from src.cropguard.detection.inference import (
    predict_pests,
    render_pest_detections,
)
from src.cropguard.detection.models import (
    get_detector_metadata,
    is_ip102_finetuned,
    load_pest_detector,
)
from src.cropguard.detection.train import (
    audit_ip102_dataset_availability,
    main as train_main,
    parse_args,
    run_training_preflight,
)


def test_voc_to_yolo_conversion_math():
    """Verifies coordinate conversion and clipping from VOC to YOLO format."""
    # Test box at center of 640x480 image
    # [160, 120, 480, 360] -> width=320, height=240 -> center=(320, 240)
    xc, yc, w, h = voc_bbox_to_yolo(160, 120, 480, 360, 640, 480)
    assert pytest.approx(xc, rel=1e-3) == 0.5
    assert pytest.approx(yc, rel=1e-3) == 0.5
    assert pytest.approx(w, rel=1e-3) == 0.5
    assert pytest.approx(h, rel=1e-3) == 0.5

    # Test coordinate clipping when box exceeds image boundaries
    xc_clip, yc_clip, w_clip, h_clip = voc_bbox_to_yolo(-10, -5, 700, 500, 640, 480)
    assert xc_clip == 0.5
    assert yc_clip == 0.5
    assert w_clip == 1.0
    assert h_clip == 1.0

    # Test rejection of degenerate/inverted bounding boxes
    with pytest.raises(ValueError, match="Degenerate bounding box"):
        voc_bbox_to_yolo(100, 100, 50, 200, 640, 480)

    with pytest.raises(ValueError, match="Invalid image dimensions"):
        voc_bbox_to_yolo(10, 10, 50, 50, 0, 480)


def test_parse_voc_xml_with_mapping(tmp_path):
    """Verifies VOC XML parsing and class mapping."""
    xml_file = tmp_path / "sample_pest.xml"
    create_synthetic_voc_sample(xml_file, img_width=640, img_height=480)
    assert xml_file.exists()

    # Parse with default behavior (integer labels)
    boxes = parse_voc_xml(xml_file)
    assert len(boxes) == 2
    # Verify first box: class 12, bounds [64, 48, 320, 240]
    cls_id, xc, yc, w, h = boxes[0]
    assert cls_id == 12
    assert pytest.approx(xc, rel=1e-2) == 0.30
    assert pytest.approx(yc, rel=1e-2) == 0.30
    assert pytest.approx(w, rel=1e-2) == 0.40
    assert pytest.approx(h, rel=1e-2) == 0.40

    # Parse with custom mapping dict
    mapping = {"12": 0, "45": 1}
    mapped_boxes = parse_voc_xml(xml_file, class_mapping=mapping)
    assert mapped_boxes[0][0] == 0
    assert mapped_boxes[1][0] == 1


def test_parse_voc_xml_malformed_multiroot_recovery(tmp_path):
    """Verifies safe recovery and deduplication from malformed multi-root XML."""
    malformed_xml = tmp_path / "multi_root_sample.xml"
    # Duplicate root elements with identical content
    block = """<annotation>
    <size><width>400</width><height>300</height><depth>3</depth></size>
    <object>
        <name>15</name>
        <bndbox><xmin>40</xmin><ymin>30</ymin><xmax>200</xmax><ymax>150</ymax></bndbox>
    </object>
</annotation>
"""
    malformed_xml.write_text(block + block, encoding="utf-8")

    rec_info = {}
    boxes = parse_voc_xml(malformed_xml, recovery_info=rec_info)
    assert len(boxes) == 1  # Deduplication of identical duplicate root
    assert boxes[0][0] == 15
    assert pytest.approx(boxes[0][1], rel=1e-3) == 0.30  # (40 + 200) / 2 / 400 = 120/400 = 0.3
    assert pytest.approx(boxes[0][2], rel=1e-3) == 0.30  # (30 + 150) / 2 / 300 = 90/300 = 0.3
    assert pytest.approx(boxes[0][3], rel=1e-3) == 0.40  # 160 / 400 = 0.4
    assert pytest.approx(boxes[0][4], rel=1e-3) == 0.40  # 120 / 300 = 0.4
    assert rec_info["recovered"] is True
    assert rec_info["blocks_found"] == 2


def test_parse_voc_xml_ip087000986_real_recovery():
    """Verifies that the actual malformed IP087000986.xml is recovered correctly."""
    ip087_path = PROJECT_ROOT / "data" / "raw" / "ip102" / "Annotations" / "IP087000986.xml"
    if not ip087_path.exists():
        pytest.skip("IP087000986.xml not present in local raw dataset")

    rec_info = {}
    boxes = parse_voc_xml(ip087_path, recovery_info=rec_info)
    assert len(boxes) == 1
    cls_id, xc, yc, w, h = boxes[0]
    assert cls_id == 86
    assert pytest.approx(xc, abs=1e-5) == 421 / 784
    assert pytest.approx(yc, abs=1e-5) == 269 / 560
    assert pytest.approx(w, abs=1e-5) == 355 / 392
    assert pytest.approx(h, abs=1e-5) == 147 / 280
    assert rec_info["recovered"] is True
    assert "junk after document element" in rec_info["original_error"]


def test_ip102_taxonomy_and_classes():
    """Verifies that IP102 class list contains all 102 official categories."""
    classes = load_ip102_classes()
    assert len(classes) == 102
    assert 1 in classes
    assert 102 in classes
    assert classes[1] == "rice leaf roller"
    assert classes[102] == "Cicadellidae"

    class_list = get_ip102_class_list()
    assert len(class_list) == 102

    # Formatting test
    assert format_pest_name("rice leaf roller") == "Rice Leaf Roller"
    assert format_pest_name("Chlumetia transversa") == "Chlumetia Transversa"


def test_yolo_model_loading_and_metadata():
    """Verifies that both base YOLO26n and fine-tuned production checkpoints load and extract metadata."""
    # Test base weights
    base_model = load_pest_detector(weights_path=PROJECT_ROOT / "yolo26n.pt", device="cpu")
    base_meta = get_detector_metadata(base_model)
    assert base_meta["model_type"] == "YOLO26n"
    assert base_meta["num_classes"] == 80  # Base COCO has 80 classes
    assert base_meta["is_ip102_finetuned"] is False
    assert "COCO" in base_meta["class_ontology"]
    assert base_meta["parameter_count"] > 2_000_000

    # Test default loader loads production fine-tuned weights
    prod_model = load_pest_detector(device="cpu")
    prod_meta = get_detector_metadata(prod_model)
    assert prod_meta["model_type"] == "YOLO26n"
    assert prod_meta["num_classes"] == 102  # IP102 fine-tuned has 102 classes
    assert prod_meta["is_ip102_finetuned"] is True
    assert "IP102" in prod_meta["class_ontology"]
    assert prod_meta["parameter_count"] > 2_000_000


def test_pest_inference_empty_detections():
    """Verifies that inference on a plain image produces zero false detections."""
    model = load_pest_detector(device="cpu")
    # Pure white synthetic image
    blank_img = Image.new("RGB", (640, 640), color=(255, 255, 255))

    res = predict_pests(model, blank_img, conf_threshold=0.25)
    assert res["num_detections"] == 0
    assert len(res["detections"]) == 0
    assert "No insect pests detected" in res["status_message"]
    assert res["latency_ms"] >= 0.0


def test_pest_overlay_rendering():
    """Verifies bounding box rendering and visual overlays."""
    img = Image.new("RGB", (400, 300), color=(100, 150, 100))

    # Test rendering with 0 detections
    rendered_empty = render_pest_detections(img, [])
    assert rendered_empty.size == (400, 300)
    assert rendered_empty.mode == "RGB"

    # Test rendering with mock detections
    mock_detections = [
        {
            "box_xyxy": [50.0, 40.0, 200.0, 180.0],
            "display_name": "Rice Leaf Roller",
            "confidence_pct": 89.4,
        }
    ]
    rendered = render_pest_detections(img, mock_detections, box_color=(220, 38, 38))
    assert rendered.size == (400, 300)
    assert rendered.mode == "RGB"
    # Verify pixels inside box or border changed from original
    orig_arr = np.array(img)
    rendered_arr = np.array(rendered)
    assert not np.array_equal(orig_arr, rendered_arr)


def test_training_cli_dry_run():
    """Verifies that the CLI dry-run executes cleanly and reports IP102 status."""
    # Test --dry-run
    exit_code = train_main(["--dry-run"])
    assert exit_code == 0


def test_training_cli_missing_dataset_safety():
    """Verifies that training exits cleanly (code 1) when dataset is missing."""
    exit_code = train_main(["--data", "non_existent_dataset.yaml"])
    assert exit_code == 1


def test_ip102_split_integrity_and_quarantine():
    """Verifies that IP102 splits are disjoint, test is quarantined, and unannotated JPEGs are excluded."""
    from scripts.prepare_ip102 import generate_ip102_splits

    summary = generate_ip102_splits()
    assert summary["status"] == "SUCCESS"
    assert summary["train_samples"] == 12898
    assert summary["val_samples"] == 2280
    assert summary["test_samples"] == 3798
    assert summary["splits_disjointness"]["train_val_disjoint"] is True
    assert summary["splits_disjointness"]["test_strictly_quarantined"] is True
    assert summary["splits_disjointness"]["unannotated_5_jpegs_excluded"] is True


def test_smoke_test_artifacts_and_metrics():
    """Verifies that smoke-test produces valid checkpoints and metrics."""
    smoke_dir = PROJECT_ROOT / "models" / "pest_detection" / "smoke_test" / "yolo26n"
    best_pt = smoke_dir / "weights" / "best.pt"
    config_json = smoke_dir / "config.json"
    metrics_json = smoke_dir / "validation_metrics.json"

    assert best_pt.exists()
    assert config_json.exists()
    assert metrics_json.exists()

    with open(metrics_json) as f:
        metrics = json.load(f)

    assert "mAP50" in metrics["metrics"]
    assert metrics["test_partition_status"] == "STRICTLY_HELD_OUT_UNTOUCHED"
    assert metrics["validation_samples_evaluated"] == 32


def test_yolo_production_training_artifacts_and_metrics():
    """Verifies that full production YOLO26n training produced expected weights and logs."""
    prod_dir = PROJECT_ROOT / "models" / "pest_detection" / "production" / "yolo26n"
    best_pt = prod_dir / "weights" / "best.pt"
    config_json = prod_dir / "config.json"
    metrics_json = prod_dir / "validation_metrics.json"
    results_csv = prod_dir / "results.csv"

    assert best_pt.exists(), "best.pt must exist in production weights"
    assert config_json.exists(), "config.json must exist in production dir"
    assert metrics_json.exists(), "validation_metrics.json must exist in production dir"
    assert results_csv.exists(), "results.csv must exist in production dir"

    with open(metrics_json) as f:
        metrics = json.load(f)

    assert metrics["validation_samples_evaluated"] == 2280
    assert metrics["metrics"]["mAP50"] >= 0.30
    assert metrics["metrics"]["precision"] >= 0.40


def test_yolo_heldout_test_metrics_and_isolation():
    """Verifies that held-out test evaluation completed on all 3,798 test samples with zero leakage."""
    report_json = PROJECT_ROOT / "reports" / "final_evaluation" / "yolo26n_test_metrics.json"
    per_class_csv = PROJECT_ROOT / "reports" / "final_evaluation" / "yolo26n_test_per_class_metrics.csv"
    sample_dir = PROJECT_ROOT / "reports" / "final_evaluation" / "yolo26n_sample_predictions"

    assert report_json.exists(), "yolo26n_test_metrics.json must exist"
    assert per_class_csv.exists(), "yolo26n_test_per_class_metrics.csv must exist"
    assert sample_dir.exists(), "Sample predictions directory must exist"

    with open(report_json) as f:
        report = json.load(f)

    assert report["test_samples_evaluated"] == 3798
    assert report["isolation_status"]["test_strictly_held_out"] is True
    assert report["isolation_status"]["train_test_overlap"] == 0
    assert report["isolation_status"]["val_test_overlap"] == 0
    assert 0.0 < report["metrics"]["map50"] <= 1.0
    assert 0.0 < report["metrics"]["precision"] <= 1.0
    assert 0.0 < report["metrics"]["recall"] <= 1.0
    assert report["hardware_inference_profile"]["latency_ms"]["inference_ms"] > 0.0


def test_verify_test_isolation_leakage_detection(tmp_path):
    """Verifies that verify_test_isolation detects artificial test data leakage."""
    from src.cropguard.detection.evaluate_heldout import verify_test_isolation

    train_txt = tmp_path / "train.txt"
    val_txt = tmp_path / "val.txt"
    test_txt = tmp_path / "test.txt"

    train_txt.write_text("img_001.jpg\nimg_002.jpg\n")
    val_txt.write_text("img_003.jpg\n")
    test_txt.write_text("img_001.jpg\nimg_004.jpg\n")  # img_001 overlaps with train!

    with pytest.raises(ValueError, match="CRITICAL: Test partition data leakage detected"):
        verify_test_isolation(train_txt, val_txt, test_txt)

