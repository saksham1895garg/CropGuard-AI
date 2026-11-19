"""
CropGuard AI - Classification Preprocessing & Imbalance Handling.
Implements architecture-specific image preprocessing pipelines and
inverse-frequency loss weight calculations for foliar disease classification.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_plantvillage_classes(csv_path: Optional[Union[str, Path]] = None) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Retrieves the canonical sorted list of 38 PlantVillage foliar disease classes
    and generates bidirectional label-index mappings.

    Args:
        csv_path: Optional path to split or manifest CSV. If None, defaults to group-aware train.csv.

    Returns:
        Tuple of (class_names, class_to_idx, idx_to_class).
    """
    if csv_path is None:
        csv_file = PROJECT_ROOT / "data" / "splits" / "classification_group_aware" / "train.csv"
        if not csv_file.exists():
            csv_file = PROJECT_ROOT / "data" / "metadata" / "dataset_manifest.csv"
    else:
        csv_file = Path(csv_path)
        if not csv_file.is_absolute():
            csv_file = PROJECT_ROOT / csv_file

    if not csv_file.exists():
        raise FileNotFoundError(f"Class source file not found: {csv_file}")

    classes = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("full_class") or row.get("class_name")
            if cls:
                classes.add(cls)

    sorted_classes = sorted(list(classes))
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
    idx_to_class = {idx: name for idx, name in enumerate(sorted_classes)}
    return sorted_classes, class_to_idx, idx_to_class


def compute_class_weights(
    csv_path: Union[str, Path],
    class_to_idx: Optional[Dict[str, int]] = None
) -> Dict[int, float]:
    """
    Computes inverse-frequency class weights exclusively from the training partition:
        w_k = N / (K * N_k)

    where:
        N  = Total training samples across all classes.
        K  = Number of distinct classes (38).
        N_k = Number of training samples belonging to class k.

    This ensures minority classes (e.g. Potato Healthy, N_k = 100) receive
    higher loss penalties during backpropagation, compensating for majority classes
    (e.g. Orange Huanglongbing, N_k = 3680) and preventing trivial majority-class bias.

    Args:
        csv_path: Path to the training partition CSV.
        class_to_idx: Optional dictionary mapping class names to integer IDs.

    Returns:
        Dictionary mapping integer class IDs to positive float loss weights.
    """
    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = PROJECT_ROOT / csv_file

    if not csv_file.exists():
        raise FileNotFoundError(f"Training split file not found: {csv_file}")

    if class_to_idx is None:
        _, class_to_idx, _ = get_plantvillage_classes(csv_file)

    counts = Counter()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row.get("full_class") or row.get("class_name")
            if cls:
                counts[cls] += 1

    total_samples = sum(counts.values())
    num_classes = len(class_to_idx)

    class_weights = {}
    for cls_name, cls_idx in class_to_idx.items():
        count = counts[cls_name]
        if count == 0:
            raise ValueError(f"Class '{cls_name}' has 0 samples in training partition {csv_file}!")
        class_weights[cls_idx] = float(total_samples / (num_classes * count))

    return class_weights


def get_model_preprocessing(model_name: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Returns the exact mathematical preprocessing function required by each architecture.

    Rules (ADR-011):
    1. 'custom_cnn':
       Scales raw uint8/float32 pixels [0, 255] linearly to [0.0, 1.0].
    2. 'mobilenet_v2':
       Applies MobileNetV2 normalization: maps [0, 255] to [-1.0, 1.0]
       via (x / 127.5) - 1.0.
    3. 'resnet50':
       Applies ResNet50 normalization: converts RGB to BGR and subtracts
       ImageNet channel means [103.939, 116.779, 123.68] from [0, 255] values.

    Args:
        model_name: One of 'custom_cnn', 'mobilenet_v2', 'resnet50'.

    Returns:
        Callable taking a float32 Tensor in [0, 255] and returning preprocessed Tensor.
    """
    name = model_name.strip().lower()

    if name in ("custom_cnn", "custom"):
        def _custom_preprocess(img: tf.Tensor) -> tf.Tensor:
            return img / 255.0
        return _custom_preprocess

    elif name in ("mobilenet_v2", "mobilenetv2", "mobilenet"):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess
        return mb_preprocess

    elif name in ("resnet50", "resnet"):
        from tensorflow.keras.applications.resnet50 import preprocess_input as rn_preprocess
        return rn_preprocess

    else:
        raise ValueError(
            f"Unknown model_name: '{model_name}'. "
            "Supported options: 'custom_cnn', 'mobilenet_v2', 'resnet50'."
        )
