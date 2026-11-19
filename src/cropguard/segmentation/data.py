"""
CropGuard AI - PlantSeg tf.data Pipeline & Manifest Loader
Implements high-throughput, memory-bounded tf.data pipelines for foliar lesion segmentation.
Supports loading directly from Plantseg.zip via safe generators or from local disk paths.
Adheres strictly to ADR-006 (cache_mode=None default to prevent host RAM overflow).
"""

import csv
import io
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import tensorflow as tf

from src.cropguard.segmentation.preprocessing import preprocess_image_and_mask


def load_segmentation_records(csv_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Loads segmentation split manifest CSV into a list of record dicts.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Segmentation split CSV not found: {p}")

    records = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def build_segmentation_dataset(
    records: List[Dict[str, Any]],
    zip_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    target_shape: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    seed: int = 42,
    subset_count: Optional[int] = None,
    cache_mode: Optional[str] = None
) -> tf.data.Dataset:
    """
    Constructs a tf.data.Dataset yielding (images, masks) batches:
    - images: float32 tensor of shape (batch, target_shape[0], target_shape[1], 3) in [0.0, 1.0]
    - masks: float32 tensor of shape (batch, target_shape[0], target_shape[1], 1) in {0.0, 1.0}

    Args:
        records: List of record dictionaries containing path information.
        zip_path: Path to Plantseg.zip (if loading directly from archive).
        batch_size: Mini-batch size.
        target_shape: (height, width) spatial resolution.
        shuffle: Whether to shuffle records.
        seed: Random seed for shuffling.
        subset_count: Optional maximum number of samples to take (e.g. for quick smoke tests).
        cache_mode: Optional caching ('disk', 'memory', or None). Default None per ADR-006.
    """
    dataset_records = list(records)

    # Optional subsetting
    if subset_count is not None and subset_count < len(dataset_records):
        rng = random.Random(seed)
        dataset_records = rng.sample(dataset_records, subset_count)

    # Determine whether loading from zip or disk
    use_zip = zip_path is not None and Path(zip_path).exists()
    zip_path_str = str(zip_path) if use_zip else None

    # Check if local paths exist
    has_local = False
    if dataset_records:
        first_r = dataset_records[0]
        if "local_image_path" in first_r and "local_mask_path" in first_r:
            has_local = Path(first_r["local_image_path"]).exists()

    def generator():
        # Shuffled generator
        indices = list(range(len(dataset_records)))
        if shuffle:
            random.Random(seed).shuffle(indices)

        zf = None
        if use_zip and not has_local:
            zf = zipfile.ZipFile(zip_path_str, "r")

        try:
            for idx in indices:
                rec = dataset_records[idx]
                if has_local:
                    img_path = Path(rec["local_image_path"])
                    mask_path = Path(rec["local_mask_path"])
                    with Image.open(img_path) as im:
                        img_pil = im.convert("RGB")
                    with Image.open(mask_path) as ma:
                        mask_pil = ma.copy()
                elif zf is not None:
                    img_path = rec.get("zip_image_path", "")
                    mask_path = rec.get("zip_mask_path", "")
                    with zf.open(img_path) as fi:
                        img_pil = Image.open(fi).convert("RGB")
                    with zf.open(mask_path) as fm:
                        mask_pil = Image.open(fm).copy()
                else:
                    raise RuntimeError("Neither valid local paths nor valid zip_path provided.")

                # Dual preprocessing
                img_arr, mask_arr = preprocess_image_and_mask(
                    img_pil, mask_pil, target_shape=target_shape
                )
                yield img_arr, mask_arr
        finally:
            if zf is not None:
                zf.close()

    output_signature = (
        tf.TensorSpec(shape=(target_shape[0], target_shape[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(target_shape[0], target_shape[1], 1), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if cache_mode == "memory":
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
