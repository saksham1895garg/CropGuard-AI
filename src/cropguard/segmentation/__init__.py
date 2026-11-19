"""
CropGuard AI - Foliar Lesion Segmentation Subsystem
Package exports for U-Net architecture, preprocessing, tf.data loading,
loss functions, callbacks, and evaluation metrics.
"""

from src.cropguard.segmentation.models import build_unet
from src.cropguard.segmentation.preprocessing import (
    normalize_image,
    binarize_mask,
    preprocess_image_and_mask,
    preprocess_image,
)
from src.cropguard.segmentation.data import (
    load_segmentation_records,
    build_segmentation_dataset,
)
from src.cropguard.segmentation.training import (
    dice_loss,
    bce_dice_loss,
    BCEDiceLoss,
    get_segmentation_callbacks,
)
from src.cropguard.segmentation.evaluation import (
    dice_coefficient,
    iou_score,
    dice_coefficient_np,
    iou_score_np,
    compute_segmentation_metrics,
    compute_lesion_severity,
)

__all__ = [
    "build_unet",
    "normalize_image",
    "binarize_mask",
    "preprocess_image_and_mask",
    "preprocess_image",
    "load_segmentation_records",
    "build_segmentation_dataset",
    "dice_loss",
    "bce_dice_loss",
    "BCEDiceLoss",
    "get_segmentation_callbacks",
    "dice_coefficient",
    "iou_score",
    "dice_coefficient_np",
    "iou_score_np",
    "compute_segmentation_metrics",
    "compute_lesion_severity",
]
