# CropGuard AI — Comprehensive Data Strategy

**Document Version:** 1.0.0  
**Effective Date:** November 17, 2026  
**Primary Maintainer:** Saksham Garg  
**Related Documents:** [DATASET_AUDIT.md](file:///home/saksham/CropGuard-AI/docs/DATASET_AUDIT.md), [DATA_PIPELINE.md](file:///home/saksham/CropGuard-AI/docs/DATA_PIPELINE.md), [ARCHITECTURE.md](file:///home/saksham/CropGuard-AI/docs/ARCHITECTURE.md)

---

## 1. Executive Summary & Vision

CropGuard AI is engineered to provide multi-modal computer vision intelligence for smart agriculture. To deliver dependable, clinically meaningful diagnostic support, the system utilizes a **three-tier vision architecture**:
1. **Foliar Disease Classification**: Identifying the specific pathogen family across 38 distinct crop-disease categories.
2. **Pathology Semantic Segmentation**: Measuring necrosis/chlorosis surface area at the pixel level to compute precise disease severity ratios.
3. **Pest & Insect Object Detection**: Localizing predatory and defoliating insects for targeted integrated pest management (IPM).

This document formalizes the data governance, ingestion protocols, quality gates, leak-free partitioning rules, and domain-adaptation strategies required to transition models from controlled lab settings to messy, unstructured agricultural fields.

---

## 2. Multi-Modal Dataset Portfolio

```
                                 CropGuard AI Data Hub
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
   PlantVillage                       PlantSeg                            IP102
(Disease Classification)        (Lesion Segmentation)                (Pest Detection)
 • 54,305 color images           • 1.57 GiB archive (Zenodo 16665770) • 102 insect classes
 • 38 crop-disease classes       • 11,458 paired masks               • PASCAL VOC XML annotations
 • 256x256 uniform               • Foliar necrosis boundaries        • Converted to normalized YOLO
 • License: CC-BY-SA-3.0         • License: CC-BY-4.0 (permissive)   • License: Academic Research
```

### 2.1 Modality Breakdown & Specifications

| Dimension | Tier 1: Classification | Tier 2: Segmentation | Tier 3: Detection |
| :--- | :--- | :--- | :--- |
| **Primary Dataset** | PlantVillage (Mohanty et al., 2016) | PlantSeg (Nature Sci. Data, 2025) | IP102 (Wu et al., CVPR 2019) |
| **Input Modality** | Detached leaf RGB photography | Leaf RGB + Binary/Semantic Mask | Multi-scale insect imagery |
| **Target Representation**| One-hot categorical scalar ($y \in \{0, \dots, 37\}$) | Spatial boolean matrix ($M \in \{0, 1\}^{H \times W}$) | Bounding boxes ($\mathbf{b} \in [0, 1]^4$) + class ID |
| **Total Volume** | 54,305 verified color images | 11,458 paired images/masks (1.57 GiB) | ~75,000 images, 102 pest classes |
| **Target Crops** | Solanaceae (Tomato, Potato, Pepper), Corn, Apple, etc. | Foliar crops, multiple plant species | Rice, corn, wheat, canola, fruit pests |
| **Primary Metric** | Macro F1-Score, Balanced Accuracy | Mean IoU (Jaccard), Dice / F1 | Mean Average Precision (mAP@50, mAP@50:95) |

---

## 3. Directory Layout & Storage Hierarchy

All raw data, processed images, manifests, and splits are structured under `data/`. Raw archives and uncompressed images are strictly excluded from version control via `.gitignore`.

```
CropGuard-AI/
└── data/
    ├── raw/                                 # [GITIGNORED] Immutable upstream downloads
    │   ├── plantvillage/
    │   │   ├── data.zip                     # 2.08 GB raw archive (CC-BY-SA-3.0)
    │   │   └── official_metadata/           # leaf-map.json, color_train.txt, color_test.txt
    │   ├── plantseg/
    │   │   └── Plantseg.zip                 # 1.57 GiB verified v6 archive (CC-BY-4.0)
    │   └── ip102/                           # Multi-part tar files / raw images
    │
    ├── processed/                           # Standardized formats ready for training
    │   ├── classification/
    │   │   └── plantvillage/                # 54,305 verified 256x256 JPEGs in 38 folders
    │   ├── segmentation/
    │   │   └── plantseg/                    # 11,458 paired images/ and masks/
    │   └── detection/
    │       └── ip102/
    │           ├── images/
    │           └── labels/                  # YOLO format text files: <cls> <xc> <yc> <w> <h>
    │
    ├── splits/                              # Deterministic, leak-free partition files
    │   ├── classification_group_aware/      # [PRIMARY] Group-aware splits (0% leaf leakage)
    │   │   ├── train.csv / train.txt        # 35,221 samples (64.86%, 12,908 leaf groups)
    │   │   ├── val.csv / val.txt            # 8,085 samples (14.89%, 2,989 leaf groups)
    │   │   ├── test.csv / test.txt          # 10,999 samples (20.25%, 4,118 leaf groups)
    │   │   └── split_summary.json           # Detailed group and class distribution
    │   ├── classification/                  # [LEGACY] Preserved naive 70/15/15 splits
    │   │   ├── train.csv / train.txt        # 38,012 samples (70.00%)
    │   │   ├── val.csv / val.txt            # 8,146 samples (15.00%)
    │   │   ├── test.csv / test.txt          # 8,147 samples (15.00%)
    │   │   └── split_summary.json           # Class-stratified audit report
    │   ├── segmentation/                    # Official author splits from Metadata.csv
    │   │   ├── train.csv (7,916 samples, 69.09%)
    │   │   ├── val.csv   (1,247 samples, 10.88%)
    │   │   └── test.csv  (2,295 samples, 20.03%)
    │   └── detection/
    │       ├── train.txt / val.txt / test.txt
    │
    └── metadata/                            # Diagnostic indices and data quality reports
        ├── dataset_manifest.csv             # Full metadata: paths, MD5, dimensions, labels
        ├── plantseg_manifest.csv            # 11,458 paired segmentation records
        ├── leaf_leakage_audit_report.json   # Full duplicate and leaf leakage report
        ├── class_distribution.csv           # Class frequencies and percentages
        └── data_quality_report.json         # Automated integrity and imbalance metrics
```

---

## 4. Rigorous Quality Gates & Validation Rules

Before any sample enters the training pipeline, it must pass five automated quality gates implemented in `scripts/prepare_*.py`:

1. **Header Verification & Corruption Check**:  
   Every image is opened and audited with `Image.verify()` and fully decoded. Zero corrupt files permitted.
2. **Spatial Dimension Audit**:  
   Input dimensions are verified. PlantVillage images are verified to be strictly $256 \times 256$ pixels.
3. **Channel Uniformity**:  
   Color mode must be 3-channel RGB. Rare RGBA images (1 in 54,305) are dynamically converted to RGB without loss of spatial information.
4. **MD5 Deduplication**:  
   File content hashes are computed to detect duplicate uploads across different directories.
5. **Mask-Image Dimension Parity**:  
   For segmentation, every mask must have exact 1-to-1 spatial parity with its parent image ($W_{mask} = W_{img}$ and $H_{mask} = H_{img}$).

---

## 5. Leak-Free Stratified Splitting Protocol

Data leakage is the most prevalent source of over-optimistic benchmark results in medical and agricultural deep learning. 

### 5.1 Stratification Rules
Given class sample counts $N_k$ for $k \in \{1, \dots, K\}$:
$$\text{Train}_k = \text{round}(0.70 \cdot N_k), \quad \text{Val}_k = \text{round}(0.15 \cdot N_k), \quad \text{Test}_k = N_k - \text{Train}_k - \text{Val}_k$$
This guarantees that minority classes (such as `Potato___healthy` with only 152 samples) maintain identical proportions across all splits ($\approx 106$ train, $23$ val, $23$ test), preventing validation bankruptcy.

### 5.2 Mathematical Proof of Zero File-Path Leakage
Let $\mathcal{S}_{\text{train}}$, $\mathcal{S}_{\text{val}}$, and $\mathcal{S}_{\text{test}}$ be the sets of unique relative file paths.
The split generator enforces:
$$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
$$\mathcal{S}_{\text{train}} \cup \mathcal{S}_{\text{val}} \cup \mathcal{S}_{\text{test}} = \mathcal{S}_{\text{manifest}}$$
This condition is verified programmatically by unit tests in `tests/test_data_pipeline.py`.

### 5.3 Physical Leaf-Level Group-Aware Splitting Protocol
While file-path disjointness prevents duplicate image loads, it does not prevent **physical-leaf leakage**:
- **Audit Discovery**: A rigorous audit of 54,305 PlantVillage images against official author leaf metadata (`leaf-map.json` from Mohanty et al., 2016) revealed that 20,015 physical leaves were photographed across multiple frames. The naive 70/15/15 split inadvertently leaked **6,464 physical leaves (32.30%)** across partitions, and allowed 4 duplicate MD5 hashes to cross into the test set.
- **Author Split Overlap**: Analysis of Sharada Mohanty's original 2016 `color_train.txt` and `color_test.txt` revealed 227 overlapping physical leaf groups (290 train images, 243 test images) due to unmerged duplicate filenames (containing "copy").
- **Group-Aware Resolution Protocol (`scripts/audit_plantvillage_leakage.py`)**:
  1. All 290 training images sharing leaf groups with the official test partition were consolidated into Test, isolating Test completely (10,999 images, 20.25%, 4,118 physical leaves).
  2. The remaining 43,306 images (15,897 leaf groups) were partitioned at the leaf-group level into Train (35,221 images, 64.86%, 12,908 leaf groups) and Val (8,085 images, 14.89%, 2,989 leaf groups) with class stratification.
- **Mathematical Proof of Zero Physical Leaf Leakage**:
  Let $\mathcal{L}_{\text{train}}$, $\mathcal{L}_{\text{val}}$, and $\mathcal{L}_{\text{test}}$ be the sets of physical leaf identifiers:
  $$\mathcal{L}_{\text{train}} \cap \mathcal{L}_{\text{val}} = \emptyset, \quad \mathcal{L}_{\text{train}} \cap \mathcal{L}_{\text{test}} = \emptyset, \quad \mathcal{L}_{\text{val}} \cap \mathcal{L}_{\text{test}} = \emptyset$$
  $$\text{MD5}(\mathcal{S}_{\text{train}}) \cap \text{MD5}(\mathcal{S}_{\text{test}}) = \emptyset$$
  Guaranteed and verified in `tests/test_data_pipeline.py`.

---

## 6. Managing Severe Class Imbalance ($\rho = 36.23$)

In PlantVillage, class sample sizes range from **$5,507$** (*Orange Citrus Greening*) down to **$152$** (*Potato Healthy*).

### Mitigation Plan:
1. **Inverse-Frequency Loss Weighting**:
   $$w_k = \frac{N_{\text{total}}}{K \cdot N_k}$$
   Implemented in Keras model training via `class_weight` dictionary during `model.fit()`.
2. **Focal Loss Formulation**:
   For severe hard-negative mining, Focal Loss replaces cross-entropy:
   $$\mathcal{L}_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
   where $\gamma = 2.0$ downweights well-classified majority samples and forces gradients to focus on minority pathogens.
3. **Evaluation Metric Decoupling**:
   Accuracy is strictly deprecated as a primary stopping metric. Models are evaluated via **Macro F1-Score** and **Class-Stratified Recall**.

---

## 7. Mitigating the Agronomic Domain Shift (Lab to Field)

### The Problem:
PlantVillage leaves were photographed on flat, artificial laboratory backgrounds (black or gray sheets). A CNN trained naively on these images will learn to associate the artificial background with the crop, suffering severe accuracy degradation when deployed in a real farm setting with variable soils, weeds, shadows, and natural canopy lighting.

### Phase 2 Augmentation Roadmap:
1. **Photometric Randomization**:
   - Random solar exposure simulation: $\Delta \text{Brightness} \in [-0.2, +0.2]$, $\Delta \text{Contrast} \in [0.8, 1.2]$.
   - Color temperature shifting (morning vs afternoon sun).
2. **Geometric Invariance**:
   - Random horizontal/vertical flips ($p = 0.5$).
   - Random rotation ($\pm 45^\circ$) and scaling ($[0.85, 1.15]$).
3. **Context Occlusion (CutOut & CutMix)**:
   - Simulating leaf overlaps, water droplets, and partial insect bites by randomly masking $16 \times 16$ to $64 \times 64$ patches.
4. **Synthetic Background Injection**:
   - Leveraging PlantSeg segmentation masks to isolate leaf blades and composite them onto field background photographs (soil, sky, companion plants).

---

## 8. High-Throughput `tf.data` Runtime Pipeline

To maximize utilization of our NVIDIA RTX 5060 Laptop GPU, training loops utilize the TensorFlow Data API (`tf.data`):

```
Disk (NVMe) ──> tf.data.Dataset.from_tensor_slices
            ──> .map(read_and_decode, num_parallel_calls=AUTOTUNE)
            ──> .map(augment_and_normalize, num_parallel_calls=AUTOTUNE)
            ──> .cache()
            ──> .shuffle(buffer_size=2048)
            ──> .batch(batch_size=32)
            ──> .prefetch(buffer_size=AUTOTUNE) ──> GPU Tensor Core
```

Empirical throughput benchmark on the RTX 5060: **$231.7$ images/second**, ensuring the GPU compute engine never starves waiting for CPU image decoding.
