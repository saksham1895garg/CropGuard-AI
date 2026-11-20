# CropGuard AI — Evaluation Quick Reference Guide

**Course:** Deep Learning (CSAI3027P) | **Evaluation:** PBL Defense  
**Project:** CropGuard AI — Multi-Model Crop Disease and Pest Surveillance System  
**Student:** Saksham Garg  
**Evaluation Time:** 3:15 PM  

---

## 1. Project Objective

CropGuard AI addresses the **diagnostic silo problem** in digital agriculture. Existing computerized tools either only classify whole-leaf diseases OR localize pests, ignoring field realities where crops suffer concurrent foliar blights and insect infestations. 

CropGuard AI deploys a **decoupled, tri-modal architecture** that concurrently provides three actionable field outputs:
1. **Foliar Disease Classification:** Predicts 38 crop-disease conditions.
2. **Lesion Semantic Segmentation:** Delineates pixel-level necrotic borders and calculates quantitative foliar surface damage percentage.
3. **Insect Pest Detection:** Localizes insect pests with bounding boxes, predicts taxa across 102 classes, and counts active pest populations.
4. **Unified Interactive Delivery:** Production-grade Streamlit dashboard (`localhost:8501`) with isolated inference pipelines, zero-detection safety guards, and one-click JSON/PNG diagnostic export.

---

## 2. Three-Model Architecture Overview

| Dimension | Branch 1: Classification | Branch 2: Segmentation | Branch 3: Object Detection |
| :--- | :--- | :--- | :--- |
| **Model** | **ResNet50** (24.13M params) | **U-Net** (7.77M params) | **YOLO26n** (2.58M params) |
| **Framework** | TensorFlow 2.22 / Keras | TensorFlow 2.22 / Keras | PyTorch 2.15 / Ultralytics |
| **Task Paradigm** | Global Multi-Class Softmax | Dense Pixel-Level Binary Sigmoid | Anchor-Free Single-Stage Bounding Boxes |
| **Input Shape** | $224 \times 224 \times 3$ RGB | $256 \times 256 \times 3$ RGB | $640 \times 640 \times 3$ RGB (Letterboxed) |
| **Target Output** | 38 Crop-Disease Classes | Binary Lesion Mask + Damage Area % | BBoxes $[x_1, y_1, x_2, y_2]$ + 102 Pest Classes + Counts |

---

## 3. Datasets & Split Integrity (Zero-Leakage Rigor)

* **PlantVillage (Classification):** 54,305 total images across 38 classes (14 crop species, 26 diseases, 12 healthy).
  * **Split:** Train: 35,221 (64.9%) | Val: 8,085 (14.9%) | Test: 10,999 (20.3%).
  * **Critical Audit:** Standard random splitting causes **32.3% physical leaf leakage** (identical leaves photographed at multiple angles appear in both train and test). We enforced a **Group-Aware Partition**, guaranteeing 0% leaf identity leakage.
* **PlantSeg (Lesion Segmentation):** 11,458 paired image-mask samples across 115 disease categories.
  * **Split:** Train: 7,916 (69.1%) | Val: 1,247 (10.9%) | Test: 2,295 (20.0%).
  * **Integrity:** Official benchmark partition preserved; 2,295 test pairs strictly quarantined.
* **IP102 (Insect Pest Detection):** 18,976 annotated field images across 102 canonical pest categories.
  * **Split:** Train: 12,898 (68.0%) | Val: 2,280 (12.0%) | Test: 3,798 (20.0%).
  * **Integrity:** 18,976 Pascal VOC XMLs converted to YOLO format (22,283 valid boxes, 0 coordinate errors). 15,178 `trainval.txt` partitioned 85/15; all 3,798 official `test.txt` images strictly quarantined.

---

## 4. Critical Preprocessing Pipelines

* **ResNet50 Preprocessing:** **Caffe BGR Mean Subtraction**. Channels converted $(R, G, B) \to (B, G, R)$ and zero-centered by ImageNet channel means: $\mu_B = 103.939, \mu_G = 116.779, \mu_R = 123.68$.  
  *(Crucial viva note: Applying $[0, 1]$ division instead of Caffe shifts activations by multiple standard deviations and collapses test accuracy from 98.42% to $<12\%$.)*
* **U-Net Preprocessing:** Standard min-max normalization to float32 range $[0.0, 1.0]$. Ground-truth PNG masks thresholded to binary $\{0, 1\}$.
* **YOLO26n Preprocessing:** Aspect-ratio-preserving bilinear letterbox resizing to $640 \times 640$ with grey border padding ($114, 114, 114$).
* **Defensive Pipeline Validation:** EXIF orientation auto-transposition, 4-channel RGBA-to-RGB alpha stripping, dynamic range validation, and anti-aliased interpolation.

---

## 5. Training Approaches & Hyperparameters

* **ResNet50 (Transfer Learning):**
  * Initialized from ImageNet-1k weights. Two-phase transfer learning schedule:
    * **Phase 1 (10 epochs):** Convolutional base frozen; custom Dense head trained at $\text{lr} = 10^{-3}$ (Adam).
    * **Phase 2 (10 epochs):** Top 25 layers unfrozen for fine-tuning at $\text{lr} = 10^{-5}$.
  * Batch Size: 16 | Image Size: $224 \times 224$ | Loss: Categorical Cross-Entropy | Early Stopping: patience = 4.
* **U-Net (Trained from Scratch):**
  * Architecture: 4-stage contracting encoder (Conv-BN-ReLU-MaxPool) + bottleneck + 4-stage expanding decoder with skip concatenation.
  * **Compound Loss Formulation:** $\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$.  
    *(Essential to counter extreme class imbalance where lesions occupy $<5\%$ of total foliar pixels.)*
  * Epochs: 20 | Batch Size: 8 | Image Size: $256 \times 256$ | Optimizer: Adam ($\text{lr} = 10^{-4}$) | Early Stopping: patience = 5.
* **YOLO26n (Fine-Tuning):**
  * Initialized from COCO-pretrained `yolo26n.pt`. Backbone: CSPDarknet with PAN-FPN feature pyramid and anchor-free head.
  * Epochs: 20 | Batch Size: 4 | Image Size: 640 | Optimizer: AdamW/SGD (peak $\text{lr} = 0.01$ with 3-epoch linear warmup).
  * Loss: Multi-task CIoU box loss + Distribution Focal Loss (DFL) + BCE class loss | Early Stopping: patience = 5 on validation mAP@50.

---

## 6. Authoritative Final Test Results

*All metrics evaluated exactly once on strictly quarantined held-out test partitions.*

| Metric Category | ResNet50 (Classification) | U-Net (Segmentation) | YOLO26n (Pest Detection) |
| :--- | :--- | :--- | :--- |
| **Test Partition Size** | **10,999 Images** | **2,295 Image-Mask Pairs** | **3,798 Images** (96 Evaluated Classes) |
| **Primary Metric** | **Accuracy: 98.42%** | **Sørensen-Dice: 63.48%** | **mAP @ 0.50: 34.33%** |
| **Secondary Metric** | **Macro F1: 97.79%** | **Mean IoU: 46.49%** | **mAP @ 0.50:0.95: 20.79%** |
| **Reliability Metric** | **Macro ROC-AUC: 99.98%** | **Specificity: 94.04%** | **Precision: 44.31%** |
| **Sensitivity Metric** | Macro Recall: 97.81% | Recall: 56.89% (Precision: 71.79%) | Recall: 38.64% |
| **Hardware Latency** | **81.76 ms** (12.2 FPS, Batch 1) | **20.71 ms** (48.3 FPS, Batch 8) | **2.03 ms** inference / **3.16 ms** total (316.5 FPS) |

*(Latency Disclosure: Batch sizes differ across tasks—ResNet50=1 on-demand, U-Net=8 batched, YOLO26n=stream. Latencies reflect production deployment, not a synthetic homogeneous benchmark.)*

---

## 7. Key Source-Code & Artifact File Paths

* **Streamlit Application:** `app/app.py` (Unified tri-modal UI, decoupled inference, download endpoints)
* **Model Checkpoints:**
  * ResNet50: `models/classification/production/resnet50/best_model.keras`
  * U-Net: `models/segmentation/production/unet/best_model.keras`
  * YOLO26n: `models/pest_detection/production/yolo26n/weights/best.pt`
* **Configuration Files:** `models/*/config.json`
* **Final Test Evaluation Reports:** `reports/final_evaluation/*_test_metrics.json`
* **Visual Presentation Assets:** `presentation/assets/*.png`
* **Test Samples:** `test_samples/classification_and_lesions/` and `test_samples/insect_pests/`

---

## 8. Three Major Technical Limitations

1. **Controlled Studio Domain Shift:** PlantVillage images were captured under uniform laboratory backdrops. Real-world unconstrained outdoor canopies with variable sun glare require domain adaptation or background segmentation.
2. **Morphological Necrosis vs Pathogen Etiology:** U-Net segments visible necrotic and chlorotic tissue, but cell death is symptom-level: it cannot differentiate fungal necrosis from herbicide chemical burn or hail trauma.
3. **Extreme Scale Variance in Pest Detection:** YOLO26n exhibits lower recall on microscopic insects ($<15$ pixels) camouflaged against complex leaf venation in dense canopies.

---

## 9. Ten Likely Viva Questions & Concise Answers

#### Q1: Why decouple the system into three models instead of an end-to-end multi-task network?
**Answer:** Conflicting spatial receptive fields and incompatible ground truth. Whole-leaf classification needs global context ($224\times224$), lesion segmentation requires dense localized boundaries ($256\times256$), and pest detection requires multi-scale feature pyramids ($640\times640$). Decoupling also prevents cascade failures and allows modular edge execution.

#### Q2: What is the exact Caffe BGR preprocessing in ResNet50, and what happens if you use $[0, 1]$ float normalization?
**Answer:** It converts RGB to BGR and subtracts ImageNet channel means ($[103.939, 116.779, 123.68]$). If $[0, 1]$ is used, input distributions shift away from the pretrained weights' expected zero-centered range, causing the first convolutional layer activations to saturate and collapsing accuracy from 98.4% to $<12\%$.

#### Q3: How did you discover and eliminate data leakage in PlantVillage?
**Answer:** PlantVillage contains multiple photos of the same physical leaf taken from slightly different angles. Naive random splitting leaks 32.3% of these identical leaves across train and test. We engineered a **group-aware split** clustering leaf identities, ensuring all images of any leaf stay strictly in one split (0% leakage).

#### Q4: Why did you use a compound loss ($0.5\cdot\text{BCE} + 0.5\cdot\text{Dice}$) for U-Net?
**Answer:** Lesions occupy $<5\%$ of total leaf area. Standard BCE is dominated by the 95% healthy background pixels, causing under-segmentation. Sørensen-Dice loss optimizes spatial overlap symmetrically. BCE provides smooth, convex early gradients while Dice enforces boundary precision.

#### Q5: Is 34.33% mAP@50 on IP102 considered good performance?
**Answer:** Yes. IP102 is an acknowledged difficult fine-grained benchmark with 102 classes in unconstrained field imagery, severe class imbalance, and camouflaged minute targets. In literature, Faster R-CNN achieves 15–22% and standard YOLO baselines achieve 28–36%. Our 34.33% mAP is achieved with only 2.6M parameters at 316.5 FPS.

#### Q6: How does the system handle zero detections safely?
**Answer:** In YOLO26n, if no candidate bounding boxes exceed the confidence threshold ($0.25$), NMS returns an empty tensor. The application intercepts this gracefully, displaying a clean "Zero Pests Detected" banner without indexing errors or false detections.

#### Q7: What is the total memory footprint? Can this run on an edge device?
**Answer:** Total model parameters: ResNet50 (24.1M $\approx$ 96 MB) + U-Net (7.8M $\approx$ 31 MB) + YOLO26n (2.6M $\approx$ 10.5 MB) = **$<140$ MB total FP32 weights**. This easily fits into an NVIDIA Jetson Orin Nano (4–8 GB) or Raspberry Pi 5.

#### Q8: Why does U-Net have high specificity (94.04%) but moderate recall (56.89%)?
**Answer:** High specificity means the model reliably preserves healthy photosynthetic tissue and avoids false positive alarms. Moderate recall occurs because fungal lesions have diffuse, faint chlorotic halos where pixel margins are ambiguous, leading the model to make conservative boundary predictions.

#### Q9: How do you prevent orientation errors when farmers upload phone photos?
**Answer:** Mobile cameras store rotation in EXIF metadata. If unhandled, raw pixel matrices load sideways. Our pipeline inspects EXIF orientation tags, applies `ImageOps.exif_transpose`, validates RGB channels, and ensures consistent orientation prior to tensor ingestion.

#### Q10: If given another semester, what single architectural enhancement would you add?
**Answer:** A **Bayesian Cross-Attention Layer** that fuses pest detections with foliar disease probabilities. For instance, detecting whiteflies (*Bemisia tabaci*) would causally upweight the prior probability of Begomovirus (Leaf Curl Virus) in the classification and segmentation branches.
