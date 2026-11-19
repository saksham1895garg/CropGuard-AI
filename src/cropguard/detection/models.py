"""
CropGuard AI - Pest Detection Model Loader & Architecture Verification.
Provides verified model initialization for YOLO26n object detection.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_BASE_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
PRODUCTION_WEIGHTS = PROJECT_ROOT / "models" / "pest_detection" / "production" / "yolo26n" / "weights" / "best.pt"
PRODUCTION_CHECKPOINT = PROJECT_ROOT / "models" / "pest_detection" / "production" / "yolo26n" / "best.pt"


def get_preferred_device() -> str:
    """Detects if CUDA acceleration is available for PyTorch."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_ip102_finetuned(model: YOLO) -> bool:
    """
    Checks if the loaded YOLO model has been fine-tuned on the 102 IP102 pest classes
    rather than retaining generic 80-class COCO weights.
    """
    if hasattr(model, "names") and isinstance(model.names, dict):
        return len(model.names) == 102
    return False


def load_pest_detector(
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> YOLO:
    """
    Loads YOLO pest detection model.
    Checks production fine-tuned weights first, falls back to local base yolo26n.pt.

    Args:
        weights_path: Optional direct path to model checkpoint (.pt).
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Loaded Ultralytics YOLO model.
    """
    if weights_path is not None:
        target_path = Path(weights_path)
    elif PRODUCTION_WEIGHTS.exists():
        target_path = PRODUCTION_WEIGHTS
    elif PRODUCTION_CHECKPOINT.exists():
        target_path = PRODUCTION_CHECKPOINT
    elif DEFAULT_BASE_WEIGHTS.exists():
        target_path = DEFAULT_BASE_WEIGHTS
    else:
        target_path = Path("yolo26n.pt")

    dev = device if device is not None else get_preferred_device()
    model = YOLO(str(target_path))
    model.to(dev)
    return model


def get_detector_metadata(model: YOLO) -> Dict[str, Any]:
    """Extracts architectural and training metadata from loaded detector."""
    num_classes = len(model.names) if hasattr(model, "names") and isinstance(model.names, dict) else 0
    finetuned = is_ip102_finetuned(model)

    # Estimate parameter count if available
    param_count = sum(p.numel() for p in model.model.parameters()) if hasattr(model, "model") and model.model is not None else 0

    return {
        "model_type": "YOLO26n",
        "parameter_count": param_count,
        "num_classes": num_classes,
        "is_ip102_finetuned": finetuned,
        "class_ontology": "IP102 Agricultural Pests (102 classes)" if finetuned else "COCO Benchmark (80 classes)",
        "device": str(model.device),
    }
