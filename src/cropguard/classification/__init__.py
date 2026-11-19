"""
CropGuard AI - Foliar Pathology Classification Subsystem.
Provides architectures, model-specific preprocessing, training pipelines,
and evaluation suites for multi-class plant disease classification.
"""

from src.cropguard.classification.models import (
    build_custom_foliar_cnn,
    build_mobilenet_v2_classifier,
    build_resnet50_classifier,
)
from src.cropguard.classification.preprocessing import (
    compute_class_weights,
    get_model_preprocessing,
    get_plantvillage_classes,
)
from src.cropguard.classification.evaluation import (
    compute_classification_metrics,
    compute_multiclass_roc_auc,
    measure_inference_latency,
    save_confusion_matrix_plot,
)

__all__ = [
    "build_custom_foliar_cnn",
    "build_mobilenet_v2_classifier",
    "build_resnet50_classifier",
    "compute_class_weights",
    "get_model_preprocessing",
    "get_plantvillage_classes",
    "compute_classification_metrics",
    "compute_multiclass_roc_auc",
    "measure_inference_latency",
    "save_confusion_matrix_plot",
]
