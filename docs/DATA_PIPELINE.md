# CropGuard AI — Data Pipeline & Ingestion Architecture

**Document Version:** 1.0.0  
**Effective Date:** November 17, 2026  
**Primary Maintainer:** Saksham Garg  
**Related Documents:** [DATA_STRATEGY.md](file:///home/saksham/CropGuard-AI/docs/DATA_STRATEGY.md), [DATASET_AUDIT.md](file:///home/saksham/CropGuard-AI/docs/DATASET_AUDIT.md)

---

## 1. Overview & Pipeline Philosophy

The CropGuard AI Data Pipeline is designed for **reproducibility, zero data leakage, and high GPU throughput**. It decouples raw archive ingestion from model training, providing structured CSV manifests, strict quality gates, and deterministic split generators.

```
                    Raw Archives (data/raw/)
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
      prepare_plantvillage  prepare_plantseg prepare_ip102
               │               │               │
               ▼               ▼               ▼
      Processed Images      Paired Masks    YOLO Labels
               │               │               │
               ├───────────────┼───────────────┘
               │               ▼
               │        prepare_plantseg (author splits)
               │               │
               ▼               ▼
      audit_plantvillage_leakage   PlantSeg Splits (11,458 pairs)
      (official leaf metadata)
               │
      ┌────────┴──────────────────────────┐
      ▼                                   ▼
Group-Aware Splits                 Legacy Splits
(data/splits/classification_       (data/splits/classification/)
 group_aware/ - 0% leaf leakage)
      │
      ▼
Safe Production tf.data Loader (src/cropguard/data/loader.py)
(Configurable cache: None [stream], disk, or memory warning; AUTOTUNE prefetch)
```

---

## 2. Pipeline Scripts Reference

### 2.1 `scripts/prepare_plantvillage.py`
**Purpose**: Extracts color images from `data/raw/plantvillage/data.zip`, audits image integrity, records spatial dimensions and MD5 checksums, and generates metadata manifests.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/prepare_plantvillage.py [OPTIONS]
```

**Options**:
- `--crops [CROPS ...]`: Filter by crop family (e.g. `Tomato Potato Pepper_bell`). Defaults to all 38 classes.
- `--limit-per-class N`: Limit images per class (useful for fast smoke-testing).
- `--skip-extract`: Skip zip extraction if images are already present on disk.

**Outputs**:
- `data/processed/classification/plantvillage/<class_name>/<filename>`: 54,305 extracted JPEGs.
- `data/metadata/dataset_manifest.csv`: 54,305 indexed records with MD5 checksums and dimensions.
- `data/metadata/class_distribution.csv`: Class frequencies and percentage proportions.
- `data/metadata/data_quality_report.json`: High-level integrity and imbalance audit metrics.

---

### 2.2 `scripts/generate_splits.py`
**Purpose**: Generates leak-free, class-stratified 70% Train, 15% Validation, and 15% Test partitions from the dataset manifest. Enforces deterministic splitting via a fixed seed and verifies zero sample overlap.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/generate_splits.py [OPTIONS]
```

**Options**:
- `--manifest PATH`: Path to `dataset_manifest.csv` (default: `data/metadata/dataset_manifest.csv`).
- `--out-dir PATH`: Target output directory (default: `data/splits/classification/`).
- `--train-ratio FLOAT`: Training split ratio (default: `0.70`).
- `--val-ratio FLOAT`: Validation split ratio (default: `0.15`).
- `--test-ratio FLOAT`: Test split ratio (default: `0.15`).
- `--seed INT`: Fixed pseudo-random seed (default: `42`).

**Outputs**:
- `data/splits/classification/train.csv` (38,012 samples) & `train.txt`
- `data/splits/classification/val.csv` (8,146 samples) & `val.txt`
- `data/splits/classification/test.csv` (8,147 samples) & `test.txt`
- `data/splits/classification/split_summary.json`: Detailed class count per partition.

---

### 2.3 `scripts/prepare_plantseg.py`
**Purpose**: Ingests, pairs, and validates foliar lesion segmentation data from Zenodo (record 16665770). Verifies exact spatial dimension parity between RGB leaves and binary lesion masks, computes lesion coverage metrics, and generates segmentation splits.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/prepare_plantseg.py [OPTIONS]
```

**Options**:
- `--test-synthetic`: Runs offline self-test using procedural synthetic leaf and mask pairs.

**Verification Metric**:
$$\text{Coverage Severity (\%)} = \frac{\text{Lesion Pixels}}{\text{Total Image Pixels}} \times 100\%$$

---

### 2.4 `scripts/prepare_ip102.py`
**Purpose**: Converts PASCAL VOC XML annotations to normalized YOLO format for agricultural pest detection.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/prepare_ip102.py [OPTIONS]
```

**Options**:
- `--voc-dir PATH`: Directory containing VOC XML annotations (default: `data/raw/ip102/Annotations`).
- `--out-dir PATH`: Output directory for YOLO `.txt` labels (default: `data/processed/detection/ip102/labels`).
- `--test-synthetic`: Runs mathematical conversion self-test.

**Mathematical Conversion**:
$$x_c = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad y_c = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$
$$w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}$$
Output line format in `<image_name>.txt`:
`<class_id> <x_center> <y_center> <width> <height>`

---

### 2.5 `scripts/inspect_datasets.py`
**Purpose**: Diagnostic CLI utility to audit the physical status of raw archives, processed images, manifests, and split disjointness.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/inspect_datasets.py
```

---

### 2.6 `scripts/visualize_samples.py`
**Purpose**: Generates high-resolution diagnostic plots for documentation, reports, and notebooks.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/visualize_samples.py [--dist] [--grid]
```

**Outputs**:
- `docs/images/plantvillage_distribution.png`: Horizontal bar chart of sample counts across all 38 classes.
- `docs/images/plantvillage_samples_grid.png`: 4x4 visual grid of diverse crop disease leaves.

---

### 2.7 `scripts/audit_plantvillage_leakage.py`
**Purpose**: Evaluates exact duplicate images (MD5/SHA256) and physical-leaf leakage across split partitions using official author metadata (`leaf-map.json`, `color_train.txt`, `color_test.txt`). Generates corrected group-aware partitions that guarantee strictly 0% leaf leakage.

**CLI Syntax**:
```bash
/home/saksham/CropGuard-AI/.venv/bin/python scripts/audit_plantvillage_leakage.py
```

**Outputs**:
- `data/metadata/leaf_leakage_audit_report.json`: Full mathematical and empirical leakage breakdown.
- `data/splits/classification_group_aware/train.csv` & `train.txt`: 35,221 images (64.86%, 12,908 leaf groups).
- `data/splits/classification_group_aware/val.csv` & `val.txt`: 8,085 images (14.89%, 2,989 leaf groups).
- `data/splits/classification_group_aware/test.csv` & `test.txt`: 10,999 images (20.25%, 4,118 leaf groups).
- `data/splits/classification_group_aware/split_summary.json`: Class and group distribution per partition.

---

## 3. Safe & High-Performance `tf.data` Production Pipeline

Training deep CNNs on the NVIDIA RTX 5060 Laptop GPU requires an asynchronous input pipeline to prevent CPU-GPU starvation while managing system memory responsibly.

### 3.1 Memory Footprint & Caching Dilemma
- In a naive setup, calling `dataset.cache()` without arguments places decoded tensors directly into system RAM.
- For 54,305 PlantVillage images at $224 \times 224 \times 3$ normalized as `float32` ($4$ bytes per element):
  $$\text{Memory} = 54{,}305 \times 224 \times 224 \times 3 \times 4\text{ bytes} \approx 32{,}695{,}437{,}312\text{ bytes} \approx 30.45\text{ GiB}$$
- In a standard workstation with 16–32 GiB RAM, an unbounded `.cache()` call causes kernel Out-Of-Memory (OOM) eviction and process crashes.

### 3.2 Production Loader (`src/cropguard/data/loader.py`):
The production data loader provides configurable caching modes:
1. `cache_mode=None` (**Default & Recommended**): Safe memory-bounded streaming pipeline. Peak memory is strictly bounded to the active batch size.
2. `cache_mode='disk'`: File-backed caching in `data/cache/`. Preserves RAM while eliminating re-decoding overhead across epochs.
3. `cache_mode='memory'`: In-memory caching for small subsets. Emits an explicit warning if sample size exceeds 5,000 images.

```python
import os
import warnings
import tensorflow as tf
import pandas as pd

def build_dataset_from_csv(
    csv_path: str,
    batch_size: int = 32,
    target_size: tuple[int, int] = (224, 224),
    is_training: bool = True,
    cache_mode: str | None = None,
    cache_dir: str = "data/cache",
    num_classes: int = 38,
    shuffle_buffer: int = 2048,
    seed: int = 42,
) -> tf.data.Dataset:
    # 1. Parse CSV manifest and validate file existence
    df = pd.read_csv(csv_path)
    filepaths = df["local_path"].tolist()
    labels = df["label_idx"].tolist() if "label_idx" in df.columns else [0] * len(df)

    # 2. Slice file paths and one-hot encoded labels
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # 3. Parallel file reading and JPEG decoding
    def _parse(filepath, label):
        raw = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        label_one_hot = tf.one_hot(label, depth=num_classes)
        return img, label_one_hot

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Safe configurable caching
    if cache_mode == "disk":
        os.makedirs(cache_dir, exist_ok=True)
        split_name = os.path.splitext(os.path.basename(csv_path))[0]
        ds = ds.cache(os.path.join(cache_dir, f"{split_name}_cache"))
    elif cache_mode == "memory":
        if len(filepaths) > 5000:
            warnings.warn(f"In-memory caching of {len(filepaths)} samples may exceed RAM.")
        ds = ds.cache()

    # 5. Shuffling (for training)
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(filepaths), shuffle_buffer), seed=seed)

    # 6. Batching
    ds = ds.batch(batch_size)

    # 7. GPU Prefetching (overlaps CPU batch prep with GPU compute)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
```

### 3.3 Preliminary Throughput Benchmark (EXP-007 Clarification)
- **Scope**: Preliminary Input Pipeline Throughput (NVMe disk I/O, parallel JPEG decode, bilinear resizing, float32 normalization, and GPU prefetch; **not** model forward/backward compute).
- **Test Configuration**: 128 samples, batch size 16, target resolution $224 \times 224 \times 3$.
- **Elapsed Time**: $0.552$ seconds.
- **Throughput**: **$231.7$ images/second** on the RTX 5060 Laptop GPU, demonstrating that the asynchronous `tf.data` pipeline will not bottleneck model training.

---

## 4. Test Suite & Verification

The data pipeline is covered by 10 automated regression tests in `tests/test_data_pipeline.py` (part of the 14-test repository suite):
1. `test_manifest_structure_and_dimensions`: Validates manifest headers, row counts (54,305), and $256 \times 256$ spatial uniformity.
2. `test_leak_free_stratified_splits`: Proves disjointness ($\text{Train} \cap \text{Val} = \emptyset$) across legacy splits.
3. `test_group_aware_split_disjointness`: Verifies strictly 0 physical-leaf overlap across `data/splits/classification_group_aware/`.
4. `test_duplicate_leakage_audit_report`: Verifies audit report structure and MD5 duplicate isolation.
5. `test_split_reproducibility`: Proves deterministic split generation from fixed random seed.
6. `test_plantseg_official_splits_disjoint`: Verifies 100% agreement with author's official PlantSeg split (7,916 / 1,247 / 2,295).
7. `test_cache_configuration`: Tests `cache_mode=None`, `cache_mode='disk'`, and memory warning threshold.
8. `test_dataset_class_distribution`: Validates all 38 classes and checks imbalance ratio calculation ($\rho = 36.23$).
9. `test_voc_to_yolo_conversion_math`: Verifies bounding box normalization $[x_c, y_c, w, h] \in [0, 1]$ and boundary clipping.
10. `test_segmentation_pairing_and_metrics`: Validates mask dimension parity and foliar lesion coverage metrics.

Run tests via:
```bash
/home/saksham/CropGuard-AI/.venv/bin/pytest -v tests/test_data_pipeline.py
```
