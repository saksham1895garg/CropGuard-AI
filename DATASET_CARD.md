---
annotations_creators:
- crowdsourced
- expert-generated
language_creators:
- found
language:
- en
license:
- cc-by-sa-3.0
- cc-by-4.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- mohanty/PlantVillage
- zenodo/16665770 (PlantSeg)
- xpwu95/IP102
task_categories:
- image-classification
- image-segmentation
- object-detection
task_ids:
- multi-class-image-classification
- semantic-segmentation
- bounding-box-object-detection
pretty_name: CropGuard AI Multi-Modal Agricultural Vision Dataset
---

# CropGuard AI — Dataset Card

## 1. Dataset Summary

The **CropGuard AI Multi-Modal Agricultural Vision Dataset** is a unified foliar pathology and pest surveillance benchmark specifically curated for smart agriculture and deep learning coursework. It integrates three modalities:
1. **Foliar Disease Classification**: 54,305 high-resolution leaf images covering 38 distinct crop-disease classes across 14 crop families (Solanaceae, Cereals, Rosaceae, Vitaceae, etc.).
2. **Pathology Semantic Segmentation**: Paired RGB foliar imagery and pixel-level binary/semantic lesion masks to measure necrosis surface area.
3. **Pest Object Detection**: 102 insect pest classes with bounding-box annotations normalized to the YOLO standard.

---

## 2. Supported Tasks & Leaderboards

### 2.1 Disease Classification (`image-classification`)
- **Input**: Detached foliar image ($256 \times 256 \times 3$ RGB JPEG).
- **Target**: Integer class label $y \in \{0, \dots, 37\}$.
- **Primary Metric**: Macro-averaged F1-Score, Balanced Accuracy.
- **Baseline**: Convolutional Neural Network (functional foliar architecture).

### 2.2 Foliar Lesion Segmentation (`image-segmentation`)
- **Input**: Foliar RGB image ($H \times W \times 3$).
- **Target**: Binary mask ($M \in \{0, 1\}^{H \times W}$), where $1=$ necrotic lesion, $0=$ healthy tissue/background.
- **Primary Metric**: Mean Intersection-over-Union (mIoU), Dice Similarity Coefficient (F1).

### 2.3 Pest Object Detection (`object-detection`)
- **Input**: High-resolution orchard/field image.
- **Target**: Normalized bounding boxes $[x_c, y_c, w, h] \in [0, 1]^4$ with class ID $c \in \{0, \dots, 101\}$.
- **Primary Metric**: mAP@0.50, mAP@0.50:0.95.

---

## 3. Dataset Structure

### 3.1 Data Instances
A typical classification manifest record in `data/metadata/dataset_manifest.csv`:
```csv
local_path,filename,full_class,crop,condition,is_healthy,width,height,channels,mode,file_size_bytes,md5
data/processed/classification/plantvillage/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3169.JPG,00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3169.JPG,Tomato___Bacterial_spot,Tomato,Bacterial_spot,False,256,256,3,RGB,15234,e5d71a9e...
```

### 3.2 Data Splits

#### Foliar Disease Classification (PlantVillage):
- **Group-Aware Leak-Free Partitions (`data/splits/classification_group_aware/` — PRIMARY)**:
  - **Train**: 35,221 images (64.86%, 12,908 physical leaf groups)
  - **Validation**: 8,085 images (14.89%, 2,989 physical leaf groups)
  - **Test**: 10,999 images (20.25%, 4,118 physical leaf groups)
  - **Leakage Status**: Strictly 0 physical leaf groups and 0 duplicate MD5 hashes shared across any sets ($\text{LeafGroups}(\text{Train}) \cap \text{LeafGroups}(\text{Test}) = \emptyset$).
- **Legacy Stratified Partitions (`data/splits/classification/` — Historical Reference)**:
  - **Train**: 38,012 images (70.00%)
  - **Validation**: 8,146 images (15.00%)
  - **Test**: 8,147 images (15.00%)
  - *Note*: Preserved on disk for historical and methodology comparison.

#### Foliar Lesion Segmentation (PlantSeg):
- **Official Author Partitions (`data/splits/segmentation/` from `Plantseg/Metadata.csv`)**:
  - **Train**: 7,916 pairs (69.09%)
  - **Validation**: 1,247 pairs (10.88%)
  - **Test**: 2,295 pairs (20.03%)
  - **Total**: 11,458 paired RGB images and binary lesion masks.

---

## 4. Dataset Creation & Curators

### 4.1 Upstream Provenance & Attribution
- **PlantVillage**: Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060; Mohanty et al. (2016), Frontiers in Plant Science.
- **PlantSeg**: Nature Scientific Data (2025). Zenodo Record 16665770 (v6 archive, July 2025).
- **IP102**: Wu, X., et al. (2019). *IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition.* IEEE CVPR 2019.

### 4.2 Licensing & Terms of Use
- **PlantVillage**: Creative Commons Attribution-ShareAlike 3.0 Unported ([CC-BY-SA-3.0](https://creativecommons.org/licenses/by-sa/3.0/)).
- **PlantSeg**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).  
  *Audit Note*: Zenodo Record 16665770 (our working copy) is licensed under CC-BY-4.0 (permissive). A newer Zenodo release (Record 17719108) shifted to CC-BY-NC-4.0 (non-commercial restriction). We retain the CC-BY-4.0 release.
- **IP102**: Non-commercial Academic Research Use Permitted.

---

## 5. Known Biases, Risks & Limitations

### 5.1 Lab vs. Field Domain Shift (Background Bias)
PlantVillage images were collected under artificial, controlled laboratory conditions with detached leaves placed on uniform black or gray background paper. Standard CNNs trained exclusively on these images are known to learn background artifacts (e.g. paper texture, illumination angles) rather than pure foliar pathology. When deployed on in-the-wild mobile cameras, accuracy drops significantly unless counterbalanced with aggressive geometric, photometric, and synthetic background augmentations.

### 5.2 Severe Class Imbalance
The dataset exhibits an imbalance ratio of **$\rho = 36.23$**, ranging from 5,507 samples (*Orange Citrus Greening*) down to 152 samples (*Potato Healthy*). Models trained with naive cross-entropy will suffer minority recall collapse. Inverse-frequency loss weighting or Focal Loss is required.

### 5.3 Physical Leaf-Level Data Leakage in Naive Splitting
Because multiple photographs were captured of the same physical leaf, naive random splitting puts images of the same leaf into both training and test partitions. In our audit, 6,464 physical leaves (32.30%) crossed partitions under naive 70/15/15 splitting. Models evaluated on naive splits will exhibit inflated test accuracy. Always evaluate on the corrected `data/splits/classification_group_aware/` partitions.

### 5.4 Diagnostic Advisory Notice
CropGuard AI is strictly an educational and decision-support tool. It is not an automated replacement for agricultural extension officers or certified laboratory microbiological assays.
