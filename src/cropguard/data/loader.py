"""
CropGuard AI - Safe & Configurable Data Loading Engine
Constructs high-throughput tf.data pipelines for foliar disease classification.

Safety Principles (ADR-010):
- In-memory caching is DISABLED by default to prevent OOM on large datasets (54k images = ~30.5 GiB in RAM).
- Optional disk caching allows persistent caching without RAM explosion.
- Parallel JPEG decoding and resizing uses tf.data.AUTOTUNE.
- Asynchronous GPU memory prefetching keeps GPU compute engines saturated.
"""

import csv
import logging
import os
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logger = logging.getLogger("cropguard.data.loader")


def parse_image_and_label(
    filepath: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int] = (224, 224),
    preprocess_fn: Optional[Callable] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Decodes JPEG file from disk, resizes to target spatial resolution,
    and applies model-specific preprocessing (or normalizes to [0.0, 1.0] by default).
    """
    raw_bytes = tf.io.read_file(filepath)
    # decode_jpeg handles RGB conversion natively
    img = tf.io.decode_jpeg(raw_bytes, channels=3)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32)
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    else:
        img = img / 255.0
    return img, label


def load_dataset_records(csv_path: Union[str, Path]):
    """
    Reads split CSV file and returns lists of absolute filepaths, string class labels,
    and integer class IDs.
    """
    csv_file = Path(csv_path)
    if not csv_file.is_absolute():
        csv_file = PROJECT_ROOT / csv_file

    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset split file not found: {csv_file}")

    filepaths = []
    class_names = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get("local_path", "")
            full_path = PROJECT_ROOT / rel_path
            filepaths.append(str(full_path))
            class_names.append(row.get("full_class", "unknown"))

    unique_classes = sorted(list(set(class_names)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    label_indices = [class_to_idx[c] for c in class_names]

    return filepaths, label_indices, class_to_idx


def get_stratified_subset(
    filepaths: list,
    labels: list,
    target_samples: int,
    seed: int = 42
) -> tuple:
    """
    Selects a class-stratified subset of target_samples ensuring that
    all classes present in labels are represented proportionally,
    with at least 1 sample per class.
    """
    from collections import defaultdict
    import random
    rng = random.Random(seed)

    by_class = defaultdict(list)
    for fp, lbl in zip(filepaths, labels):
        by_class[lbl].append(fp)

    total_available = len(filepaths)
    if target_samples >= total_available:
        return filepaths, labels

    subset_fps = []
    subset_lbls = []

    for lbl, fps in sorted(by_class.items()):
        alloc = max(1, round(target_samples * (len(fps) / total_available)))
        alloc = min(alloc, len(fps))
        shuffled = list(fps)
        rng.shuffle(shuffled)
        chosen = shuffled[:alloc]
        subset_fps.extend(chosen)
        subset_lbls.extend([lbl] * len(chosen))

    combined = list(zip(subset_fps, subset_lbls))
    rng.shuffle(combined)
    subset_fps, subset_lbls = zip(*combined)
    return list(subset_fps), list(subset_lbls)


def build_classification_dataset(
    csv_path: Union[str, Path],
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    shuffle_buffer: int = 2048,
    cache_mode: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    seed: Optional[int] = 42,
    limit: Optional[int] = None,
    preprocess_fn: Optional[Callable] = None
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset pipeline.

    Args:
        csv_path: Path to train.csv, val.csv, or test.csv split file.
        batch_size: Mini-batch size for training/inference.
        image_size: Target (height, width) spatial dimensions.
        shuffle: Whether to shuffle samples before batching.
        shuffle_buffer: Shuffle buffer size.
        cache_mode: Caching policy:
            - None (default): Streaming mode without full caching. Safest for large datasets.
            - 'disk': Caches decoded tensors to local disk files.
            - 'memory': Caches tensors in RAM. Emits warning if dataset is large (>5,000 samples).
        cache_dir: Directory for file-backed disk caching if cache_mode == 'disk'.
        seed: Random seed for deterministic shuffling.
        limit: Optional maximum number of samples to load (useful for testing). When specified,
            extracts a class-stratified subset ensuring all classes are represented.
        preprocess_fn: Optional model-specific preprocessing function taking a float32 tensor
            in [0, 255] and returning the preprocessed tensor. If None, scales to [0, 1].

    Returns:
        tf.data.Dataset yielding batches of (images, labels).
    """
    filepaths, label_indices, class_to_idx = load_dataset_records(csv_path)

    if limit is not None and limit > 0:
        filepaths, label_indices = get_stratified_subset(
            filepaths, label_indices, target_samples=limit, seed=seed if seed is not None else 42
        )

    total_samples = len(filepaths)

    # 1. Base dataset from tensor slices
    ds = tf.data.Dataset.from_tensor_slices((filepaths, label_indices))

    # 2. Parallel image decoding and preprocessing
    def _map_fn(fp, lbl):
        return parse_image_and_label(fp, lbl, image_size=image_size, preprocess_fn=preprocess_fn)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # 3. Configurable Caching Layer (Memory-Safe by Default)
    if cache_mode == "memory":
        if total_samples > 5000:
            est_gib = (total_samples * image_size[0] * image_size[1] * 3 * 4) / (1024 ** 3)
            warnings.warn(
                f"Memory caching enabled for {total_samples:,} samples! "
                f"Estimated memory consumption is ~{est_gib:.2f} GiB. "
                "Consider setting cache_mode=None or cache_mode='disk' to avoid OOM.",
                UserWarning
            )
        ds = ds.cache()
    elif cache_mode == "disk":
        if cache_dir is None:
            cache_dir = PROJECT_ROOT / "data" / "cache"
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file_prefix = str(cache_path / f"cache_{Path(csv_path).stem}")
        ds = ds.cache(cache_file_prefix)
    elif cache_mode is None or cache_mode == "none":
        # Safe default: no caching, stream on demand
        pass
    else:
        raise ValueError(f"Unsupported cache_mode: '{cache_mode}'. Choose None, 'disk', or 'memory'.")

    # 4. Shuffling
    if shuffle:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, max(1, total_samples)), seed=seed)

    # 5. Batching
    ds = ds.batch(batch_size)

    # 6. GPU Prefetching (asynchronous pipeline)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
