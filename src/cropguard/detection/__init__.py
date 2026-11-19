"""
CropGuard AI - Foliar Insect Pest Detection Module.
Provides YOLO26n object detection and IP102 taxonomy utilities.
"""

from src.cropguard.detection.classes import (
    IP102_CLASSES_CANONICAL,
    format_pest_name,
    get_ip102_class_list,
    load_ip102_classes,
)
from src.cropguard.detection.models import (
    get_detector_metadata,
    is_ip102_finetuned,
    load_pest_detector,
)
from src.cropguard.detection.inference import (
    predict_pests,
    render_pest_detections,
)

__all__ = [
    "IP102_CLASSES_CANONICAL",
    "load_ip102_classes",
    "get_ip102_class_list",
    "format_pest_name",
    "load_pest_detector",
    "is_ip102_finetuned",
    "get_detector_metadata",
    "predict_pests",
    "render_pest_detections",
]
