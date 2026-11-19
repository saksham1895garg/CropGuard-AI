# CropGuard AI — Current Project State

**Last Updated:** November 18, 2026  
**Project Repository:** `saksham1895garg/CropGuard-AI`  
**Current Branch:** `main`  
**Primary Maintainer:** Saksham Garg  
**Development Paradigm:** Interactive Notebook-First & Documentation-First AI Collaboration  

---

## 1. Project Objective & Scope

**CropGuard AI** is an intelligent crop disease and pest surveillance decision-support system designed to assist agriculturalists and students in identifying, segmenting, and tracking crop pathogens and pests from leaf imagery.

### Core Capabilities (System Portfolio):
1. **Disease Classification:** Identification of visible foliar crop pathogens across 38 crop-disease classes (PlantVillage benchmark).
2. **Pathology Segmentation:** Pixel-level segmentation of infected necrotic and chlorotic tissue where ground-truth masks are available (PlantSeg benchmark).
3. **Pest Detection:** Bounding-box localization of common agricultural pests using a dedicated YOLO model (IP102 benchmark).
4. **Surveillance & Temporal Analysis:** Severity progression tracking and risk forecasting.
5. **Interactive Interface:** Local web dashboard (Streamlit) reporting predictions, confidence intervals, diagnostic heatmaps, and inference latency.

> [!IMPORTANT]
> **Advisory Tool Notice:** CropGuard AI is designed as an educational and decision-support tool. It is not an automated replacement for an agronomist or microbiological laboratory assay. Disease severity or progression is never claimed from an isolated static image.

---

## 2. Current Development Stage

**Current Stage:** **Milestone 15 Completed — Final YOLO26n Held-Out Test Evaluation & Full Tri-Model Dashboard Integration**  
**Status:** **Complete, Fully Evaluated & Checkpoints Permanently Frozen** (Notebooks 01–08 completed and executed by user with all outputs permanently preserved; Production classification pipeline completed with ResNet50 evaluated at **98.42% test accuracy** on 10,999 held-out images; Foliar lesion segmentation subsystem completed with U-Net evaluated at **63.48% test Dice** and **46.49% test IoU** on 2,295 held-out test pairs across 115 categories; Insect pest detection subsystem completed with Ultralytics YOLO26n [2.58M params, 102 classes] evaluated on 3,798 official held-out test images at **34.33% test mAP@50**, **20.79% test mAP@50-95**, **44.31% precision**, **38.64% recall**, and **2.03 ms latency** [316.5 FPS]; Streamlit web application live and fully operational at `app/app.py` on `localhost:8501` featuring all three models in a unified workflow with zero-detection safety cards, curated demo presets, and JSON/PNG export suites; Dual runtime [TensorFlow 2.22 + PyTorch 2.15/CUDA] operating cleanly in single virtual environment with `pip check` clean; Automated regression test suite passing at **77/77 tests** across 9 test modules).

CropGuard AI has achieved its full functional project scope: a comprehensive, explainable, three-model deep learning diagnostic platform combining 38-class crop disease classification (ResNet50), pixel-level foliar lesion segmentation (U-Net), and insect pest detection (YOLO26n) with an interactive web dashboard, diagnostic report exports, and rigorous academic disclaimers.

---

## 3. Verified Technology Stack

| Component | Confirmed Specification / Choice | Notes |
| :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu on WSL2) | Host Windows 11 kernel |
| **Language** | Python 3.14.4 | Standalone dedicated `.venv` in workspace |
| **Virtual Environment** | `/home/saksham/CropGuard-AI/.venv` | Dual-registered `cropguard-ai` ipykernel |
| **Primary DL Framework**| TensorFlow 2.22.0rc0 | With CUDA 12 runtime packages (Classification & Segmentation) |
| **Object Detection Framework**| PyTorch 2.15.0+cu132 & Ultralytics 8.4.215 | With native CUDA 13.2 GPU acceleration (YOLO26n Pest Detection) |
| **High-Level API** | Keras Nightly (`3.16.0.dev2026110903`) | Integrated with TensorFlow |
| **Numerical Computing** | NumPy 2.5.4 | C-accelerated array computing |
| **Visualization** | Matplotlib 3.11.2 | Inline notebook plotting & diagnostic figures |
| **Image Processing** | Pillow 13.0.0 | High-performance image loading & validation |
| **Testing Framework** | pytest 9.2.0 | Regression test suite: **77/77 passing tests** across 9 modules |
| **Notebook Kernel** | `cropguard-ai` (`CropGuard AI (.venv)`) | Configured in Jupyter/VS Code UI |
| **Hardware Accelerator**| NVIDIA GeForce RTX 5060 Laptop GPU | ~8 GB VRAM (8,151 MiB), Compute Capability 12.0a |
| **CUDA Driver / Runtime**| Driver: 596.21 (NVIDIA-SMI 595.58.04, Host CUDA: 13.2) / Runtime: CUDA 12 | Verified with tensor matrix multiplication |

---

## 4. Completed Modules & Artifacts

| Module / Artifact | Path | Status | Verification Detail |
| :--- | :--- | :--- | :--- |
| **Environment Verification** | `scripts/verify_environment.py` | Complete | Verified Python 3.14, TF 2.22, RTX 5060 GPU detection & matmul |
| **Notebook 01** | `notebooks/01_neurons_and_mlp.ipynb` | **Completed & Executed** | Artificial Neurons, McCulloch-Pitts, Sigmoid, Perceptron XOR, MLP forward pass. All 10 code cells executed by user with figures preserved. |
| **Notebook 02** | `notebooks/02_backpropagation_and_optimizers.ipynb` | **Completed & Executed** | Forward pass, MSE/BCE Loss, Chain Rule, Manual Backprop, Gradient Descent, SGD, Momentum, NAG, Adam benchmark. All 10 code cells executed by user with figures preserved. |
| **Notebook 03** | `notebooks/03_tensor_operations_broadcasting_and_svd.ipynb` | **Completed & Executed** | Tensors (0D-4D), Tensor arithmetic, Broadcasting rules, Cosine similarity, MatMul, Reshaping, `tf.GradientTape` auto-diff, SVD low-rank compression. All 14 code cells executed by user with figures preserved. |
| **Notebook 04** | `notebooks/04_learning_rules_and_training.ipynb` | **Completed & Executed** | Learning Rules (Hebbian, Oja's, Competitive, Delta/Adaline, Boltzmann), Pattern Association, Hopfield Memory, Function Approximation, Generalization & Overfitting. All 10 code cells executed by user with figures preserved. |
| **Notebook 05** | `notebooks/05_cnn_feature_maps_and_pooling.ipynb` | **Completed & Executed** | Convolutions, 2D kernels, Sobel edge filters, Strides, Padding (VALID vs SAME), Pooling (Max vs Avg), 3D multi-channel convolutions, Receptive Field calculus, Dense transition params, Keras functional foliar CNN, Feature Map extraction. All 14 code cells executed by user with figures preserved. |
| **Notebook 06** | `notebooks/06_agricultural_dataset_exploration.ipynb` | **Completed & Executed** | Agricultural vision tasks, PlantVillage dataset audit, Chlorophyll degradation histograms, Imbalance calculus ($\rho=36.80$), Class weighting, Multi-crop gallery, Stratified leak-free splits, PlantSeg lesion segmentation overlay, VOC-to-YOLO conversion, High-throughput `tf.data` pipeline. All 9 code cells executed by user with figures preserved. |
| **Notebook 07** | `notebooks/07_cnn_architectures_and_transfer_learning.ipynb` | **Completed & Executed** | Custom Foliar CNN vs Transfer Learning (MobileNetV2, ResNet50), Depthwise Separable Convolutions, Residual Skip Connections, Model-Specific Preprocessing, Inverse-Frequency Class Weights, GPU Inference Profiling, Diagnostic Error Analysis, Agricultural Confounding/Domain Shift, Hands-on Exercises & 10 Comprehensive Viva Questions. All 13 code cells executed interactively by user; quick pipeline smoke test figures and metrics preserved. |
| **Notebook 08** | `notebooks/08_unet_lesion_segmentation.ipynb` | **Completed & Executed** | U-Net Foliar Lesion Segmentation, Classification vs Segmentation, Contracting Encoder & Expanding Decoder, Skip Connections, Image-Mask Pairing & PlantSeg format binarization, Receptive Field calculus, Combined BCE + Dice Loss, Sørensen-Dice & IoU metrics, Lesion Severity extraction, and 10 Comprehensive Viva Questions with academic answers. All 9 code cells successfully executed interactively by user with outputs and diagnostic figures permanently preserved; PlantSeg test partition strictly quarantined. |
| **Segmentation Subsystem** | `src/cropguard/segmentation/` | Complete | Modular segmentation subsystem: `models.py` (Standard U-Net [7,771,873 params, 7,765,985 trainable]), `preprocessing.py` (image [0,1] scaling, mask binarization, spatial resizing, EXIF orientation handling), `data.py` (memory-safe streaming from Plantseg.zip with ADR-006 `cache_mode=None`), `training.py` (`BCEDiceLoss`, `dice_loss`, callbacks), `evaluation.py` (Sørensen-Dice, IoU, confusion metrics, foliar lesion severity index). |
| **Segmentation Training CLI** | `src/cropguard/segmentation/train.py` | Complete | Standalone CLI training pipeline for U-Net lesion segmentation supporting `--epochs`, `--batch-size`, `--learning-rate`, `--dry-run`, `--smoke-test`, dynamic GPU memory growth, and strict quarantine of PlantSeg test split. Smoke test verified on RTX 5060 GPU with artifacts (`best_model.keras`, `config.json`, `validation_metrics.json`, `sample_predictions/*.png`). |
| **Production U-Net Model** | `models/segmentation/production/unet/` | Complete | Full production U-Net checkpoint trained on 7,916 PlantSeg pairs (20 epochs) on RTX 5060 GPU: `best_model.keras` (90 MB), **Val Dice 62.34%**, **Val IoU 45.28%**, **Precision 71.28%**, **Recall 55.38%**, **Specificity 94.31%**, Latency 21.17 ms/image (47.2 FPS); 6 five-panel diagnostic predictions; test split strictly quarantined. |
| **PlantSeg Full Census Mask Audit** | `scripts/audit_plantseg_masks.py` | Complete | Full census of all 9,163 train and val pairs: 9,155 (99.91%) exact matching dimensions, 8 (0.09%) swapped dimensions due to iPhone EXIF tag 274: 6, 0% other mismatches. Verified raw masks encode disease IDs [1, 115] binarized cleanly via `mask > 0`. Evidence saved in `data/metadata/plantseg_mask_audit_report.json` and `docs/images/plantseg_sample_pairs.png`. Test split strictly quarantined. |
| **Classification Subsystem** | `src/cropguard/classification/` | Complete | Modular classification subsystem: `models.py` (Custom CNN [429,158 params], MobileNetV2 [2,600,806 params], ResNet50 [24,130,214 params]), `preprocessing.py` (model-specific normalization & inverse-frequency class weights), `training.py` (compilation & callbacks with TensorBoard), `evaluation.py` (pure NumPy multi-class metrics, Wilcoxon-Mann-Whitney ROC-AUC & confusion matrix plotting). |
| **Production Training CLI** | `src/cropguard/classification/train.py` | Complete | Standalone CLI training pipeline supporting `custom_cnn`, `mobilenet_v2`, `resnet50`, two-phase transfer learning with frozen BatchNorm, pure NumPy ROC-AUC, dynamic memory growth, `--dry-run` and `--eval-only` modes, and structured artifact generation. |
| **Final Test Evaluation CLI** | `src/cropguard/classification/evaluate_heldout.py` | Complete | Standalone held-out evaluation runner verifying test partition isolation, executing inference on frozen checkpoints, computing pure NumPy metrics, and generating publication-grade confusion matrix heatmaps and per-class CSVs. |
| **Final Classification Reports**| `reports/final_evaluation/` | Complete | ResNet50 held-out evaluation artifacts: `resnet50_test_metrics.json` (**98.42% accuracy**, 0.9779 Macro F1, 0.9998 ROC-AUC on 10,999 test samples), `resnet50_confusion_matrix.png`, and `resnet50_per_class_metrics.csv`. |
| **Final Segmentation Reports**  | `reports/final_evaluation/` | Complete | U-Net held-out evaluation artifacts: `unet_test_metrics.json` (**63.48% Dice**, **46.49% IoU**, **71.79% Precision**, **94.04% Specificity** on 2,295 test pairs across 115 categories), `unet_test_per_class_metrics.csv`, and 6 five-panel diagnostic predictions in `unet_sample_predictions/`. |
| **Streamlit Web Application**   | `app/app.py` | Complete | Production web dashboard combining ResNet50, U-Net, and YOLO26n with image uploader, curated presets, top-3 disease candidates, 3-panel lesion segmentation, damage assessment, insect pest bounding boxes, JSON/PNG export, and scientific disclaimer. Live on `http://localhost:8501`. |
| **Inference Engine**            | `src/cropguard/inference.py`| Complete | Decoupled production inference module with EXIF orientation correction, model-specific preprocessing, eager forward pass, severity calculus, pest detection wrappers, and overlay generation. |
| **Pest Detection Subsystem**    | `src/cropguard/detection/` | Complete | Modular pest detection subsystem: `models.py` (YOLO26n loader, CUDA device binding, metadata extraction), `inference.py` (bounding box parsing, coordinate clipping, zero-pest safety handling, high-contrast visual badge rendering), `classes.py` (canonical IP102 102-class taxonomy loader & title-case formatter). |
| **Pest Detection Training CLI** | `src/cropguard/detection/train.py` | Complete | Standalone CLI training pipeline supporting `--epochs`, `--batch-size`, `--imgsz`, `--dry-run`, `--smoke-test`, dynamic GPU memory, and empirical IP102 dataset validation. Full 20-epoch training completed on RTX 5060 GPU with checkpoints, metrics, and PR curves saved in `models/pest_detection/production/yolo26n/`. |
| **Pest Detection Held-Out CLI** | `src/cropguard/detection/evaluate_heldout.py` | Complete | Standalone held-out evaluation runner verifying test partition isolation, executing inference on frozen `best.pt`, extracting per-class AP, precision, recall, and latency metrics across all 3,798 test samples. |
| **Final Pest Detection Reports**| `reports/final_evaluation/` | Complete | YOLO26n held-out evaluation artifacts: `yolo26n_test_metrics.json` (**34.33% mAP@50**, **20.79% mAP@50-95**, **44.31% precision**, **38.64% recall**, **2.03 ms latency** on 3,798 test samples), `yolo26n_test_per_class_metrics.csv` (96 evaluated classes), and 6 diagnostic predictions in `yolo26n_sample_predictions/`. |
| **IP102 Benchmark Dataset**      | `data/raw/ip102/` & `data/processed/detection/ip102/` | Complete | Complete detection benchmark: 18,976 VOC XMLs converted to YOLO format (22,283 bounding boxes, 97 observed classes), 18,981 JPEGs, stratified 85/15 internal split (12,898 train, 2,280 val, 3,798 quarantined test), disjointness verified 100%, 5 unannotated JPEGs excluded. |
| **Application Test Suite**      | `tests/test_app.py` | Complete | 11 unit tests covering image orientation, preprocessing, prediction formatting, synthetic inference, lesion calculation, pest detection wrappers, and overlay blending. |
| **Pest Detection Test Suite**   | `tests/test_detection.py` | Complete | 15 unit tests covering VOC-to-YOLO conversion math, coordinate clipping, degenerate box rejection, VOC XML parsing with mapping, multi-root malformed XML recovery on synthetic sample and real `IP087000986.xml`, IP102 102-class taxonomy, YOLO26n model loading, zero-detection handling, visual rendering, CLI dry-run safety, stratified split disjointness, production artifacts, and held-out test evaluation. |
| **Application Screenshot**      | `docs/images/app_demo_screenshot.png` | Complete | Publication-quality 180 DPI composite dashboard rendering showing live tri-model execution on curated demo leaf. |
| **Automated Test Suite**        | `tests/` | Complete | **77/77 passing tests** across 9 test modules (`test_environment.py` [4], `test_data_pipeline.py` [10], `test_classification.py` [9], `test_cli_training.py` [9], `test_final_evaluation.py` [4], `test_segmentation.py` [8], `test_segmentation_training.py` [7], `test_app.py` [11], `test_detection.py` [15]). |
| **PlantVillage Ingestion** | `scripts/prepare_plantvillage.py` | Complete | Extracted and audited 54,305 color images, 0 corrupt files, uniform 256x256 resolution. Generated manifest, class distribution, and quality report. |
| **PlantVillage Leaf Leakage Audit** | `scripts/audit_plantvillage_leakage.py` | Complete | Discovered 32.3% leaf leakage in naive split; detected 21 duplicate MD5 hashes; generated group-aware partitions in `data/splits/classification_group_aware/` with **strictly 0% leaf leakage**. Full report in `data/metadata/leaf_leakage_audit_report.json`. |
| **Stratified Split Generator**| `scripts/generate_splits.py` | Complete | Legacy 70/15/15 partitions preserved in `data/splits/classification/`: Train (38,012), Val (8,146), Test (8,147). |
| **Group-Aware Splits** | `data/splits/classification_group_aware/` | Complete | Leak-free leaf-grouped partitions: Train (35,221 / 64.86%), Val (8,085 / 14.89%), Test (10,999 / 20.25%). Verified 0 physical leaf overlap. |
| **Safe `tf.data` Loader** | `src/cropguard/data/loader.py` | Complete | Safe configurable data loader with model-specific `preprocess_fn` parameter and class-stratified subset sampling for quick experiments (`cache_mode=None` default to prevent ~30.45 GiB RAM overflow; optional disk caching; warned memory caching; parallel AUTOTUNE & prefetch). |
| **IP102 VOC-to-YOLO & Full Conversion** | `scripts/prepare_ip102.py` | Complete | PASCAL VOC XML to normalized YOLO converter $[x_c, y_c, w, h] \in [0, 1]$ with multi-root recovery and block deduplication. 100% conversion (18,976/18,976), 1 recovered (`IP087000986.xml`), 22,283 valid bounding boxes, audit report in `data/metadata/ip102_conversion_report.json`. |
| **PlantSeg Ingestion & Audit** | `scripts/prepare_plantseg.py` | Complete | Foliar lesion segmentation pairing verification, coverage severity metric calculator, and Zenodo v6 (CC-BY-4.0) vs v7 (CC-BY-NC-4.0) licensing audit. Ingested 11,458 official author pairs. |
| **Inspection CLI** | `scripts/inspect_datasets.py` | Complete | Audits raw archives, processed images, manifests, and split disjointness. |
| **Visualizer** | `scripts/visualize_samples.py` | Complete | Generates distribution and sample gallery figures in `docs/images/`. |
| **Automated Test Suite** | `tests/` | Complete | **61/61 passing tests** across `test_environment.py` (4), `test_data_pipeline.py` (10), `test_classification.py` (9), `test_cli_training.py` (9), `test_final_evaluation.py` (4), `test_segmentation.py` (8), and `test_segmentation_training.py` (7). |
| **Documentation System** | `docs/` & `DATASET_CARD.md` | Complete | Comprehensive documentation suite: `PROJECT_STATE.md`, `DATASET_AUDIT.md`, `DATA_PIPELINE.md`, `DATA_STRATEGY.md`, `EXPERIMENT_LOG.md`, `PROGRESS_LOG.md`, `DECISIONS.md`, `NEXT_SESSION.md`, `ARCHITECTURE.md`, `COURSE_MAPPING.md`, `AGENT_HANDOFF.md`, and `DATASET_CARD.md`. |

---

## 5. Upcoming Academic Deliverables

| Deliverable | Target / Reference | Scope | Status |
| :--- | :--- | :--- | :--- |
| **Interactive Web Demo** | `app/app.py` | Live dual-pipeline Streamlit demonstration on `localhost:8501` | **Ready / Operational** |
| **Viva Voce Presentation** | Course Defense | Presentation of Notebooks 01–08, ResNet50 (98.42%), and U-Net (63.48%) | Ready |
| **Final Course Report** | Academic Submission | UPES CSAI3027P course outcome synthesis | Ready |

---

## 6. Dataset & Model Status

- **PlantVillage Dataset:** **Ingested, Audited, and Verified.** 54,305 color images stored in `data/processed/classification/plantvillage/`. Indexed in `data/metadata/dataset_manifest.csv`. Reconciled exact duplicates: 21 duplicate MD5 hashes (42 images) in total; 10 hashes leaked across partitions in naive split (4 between Train & Test). Evaluated perceptual similarity: 23 dHash groups at Hamming = 0 (46 images) with 0 crossing corrected splits; near-duplicates at Hamming 1–2 eliminated (0 cross-partition pairs). Mapped taxonomy: 41,111 author-confirmed images (75.70%, 7,593 groups) and 13,194 fallback images (24.30%, 12,422 groups), with 0 unresolved. Corrected group-aware partitions saved in `data/splits/classification_group_aware/`: Train (35,221 / 64.86%), Val (8,085 / 14.89%), Test (10,999 / 20.25%) with strictly 0 physical leaf, MD5, or dHash (Hamming <= 2) overlap. Legacy splits preserved in `data/splits/classification/`.
- **PlantSeg Dataset:** **Ingested, Audited, and Verified.** Zenodo record 16665770 (v6 archive, 1,691,076,634 bytes, MD5 `23385d877bc84bab51b54630055c75b8`). Archive-internal licenses verified in `Plantseg/Metadata.csv`: 10,325 CC0 (90.11%) and 1,133 CC-BY (9.89%), both permissive. Zenodo API package-level license status marked UNVERIFIED (external API blocked by HTTP 403). Official author splits in `data/splits/segmentation/`: Train (7,916, 69.09%), Val (1,247, 10.88%), Test (2,295, 20.03% quarantined). Full census across all 9,163 train and val pairs: 9,155 (99.91%) exact matching spatial dimensions, 8 (0.09%) swapped dimensions due to iPhone EXIF tag 274: 6 (handled via `ImageOps.exif_transpose`), 0% other mismatches. Raw masks encode disease class integer IDs [1, 115] with 0 as background, binarized cleanly to {0.0, 1.0} for foliar lesion segmentation. Empty mask percentage: 0.0%. Test partition strictly quarantined. Evidence saved in `data/metadata/plantseg_mask_audit_report.json` and `docs/images/plantseg_sample_pairs.png`.
- **IP102 Dataset:** **Ingested, Audited, 100% Converted, and Partitioned.** 18,976 VOC XML annotations converted to normalized YOLO format ($[x_c, y_c, w, h] \in [0, 1]$) with 22,283 valid bounding boxes across 97 observed classes. 1 malformed multi-root XML (`IP087000986.xml`) safely recovered without fabricating data. 18,981 JPEGs stored in `data/raw/ip102/JPEGImages/` (symlinked to `data/processed/detection/ip102/images/`). Internal stratified 85/15 train/val split created exclusively from 15,178 `trainval.txt` IDs: Train (12,898), Val (2,280), Smoke Train (64), Smoke Val (32). Official test partition (3,798 IDs) strictly quarantined and untouched. Five unannotated JPEGs excluded. Full audit reports in `data/metadata/ip102_conversion_report.json` and `data/metadata/ip102_split_summary.json`.
- **Pest Detection Model Weights:** **Full Production YOLO26n Trained, Evaluated, and Frozen.** Trained for 20 epochs on 12,898 IP102 images with RTX 5060 Laptop GPU: `models/pest_detection/production/yolo26n/weights/best.pt` (5.5 MB). Validation metrics on 2,280 images: Precision **46.96%**, Recall **39.84%**, mAP@50 **35.21%**, mAP@50-95 **21.39%**, latency 2.2 ms. Final held-out evaluation on 3,798 test images: Precision **44.31%**, Recall **38.64%**, mAP@50 **34.33%**, mAP@50-95 **20.79%**, latency **2.03 ms / image** (316.5 FPS), zero leakage verified. Artifacts saved in `reports/final_evaluation/` (`yolo26n_test_metrics.json`, `yolo26n_test_per_class_metrics.csv`, and `yolo26n_sample_predictions/`). Checkpoint frozen.
- **Trained Model Weights:** **Production ResNet50 fully evaluated (98.42% test acc, 0.9779 Macro F1, 0.9998 ROC-AUC); MobileNetV2 benchmarked (94.84% val acc); smoke-test checkpoints stored.** Checkpoints from the interactive execution of Notebook 07 are saved in `models/classification/` (`custom_cnn_best.keras` [5.0 MB], `mobilenet_v2_best.keras` [14 MB], `resnet50_best.keras` [97 MB]); history logs in `reports/experiments/*.csv`. The standalone CLI production training pipeline has completed full training for MobileNetV2 and ResNet50 (`models/classification/production/resnet50/best_model.keras` [181.3 MB]). Final held-out test evaluation completed and saved in `reports/final_evaluation/`. Model weights are strictly gitignored.
- **Data Policy:** Raw datasets (`data/raw/`), processed images, and model weights are strictly gitignored.

---

## 7. Known Technical Limitations & Constraints

1. **PTX JIT Compilation on Blackwell Architecture (RTX 5060):**
   * The NVIDIA RTX 5060 Laptop GPU possesses Compute Capability `12.0a`. TensorFlow 2.22.0rc0 pre-built binaries target compute capabilities up to `9.0`. On first execution of a new CUDA kernel, TensorFlow JIT-compiles from PTX, introducing an initial warm-up latency of ~1.5–2 seconds. Subsequent executions are instantaneous.
2. **GPU Memory Growth Requirement:**
   * Dynamic GPU memory allocation (`tf.config.experimental.set_memory_growth(gpu, True)`) is mandatory at the beginning of any notebook or script to avoid out-of-memory errors from aggressive preallocation.
3. **Severe Class Imbalance ($\rho = 36.23$):**
   * Handled via computed inverse-frequency class weights ($w_k$) or Focal Loss in training loops to prevent minority recall collapse.
4. **tf.data In-Memory Caching Footprint:**
   * Caching the entire 54,305-image PlantVillage dataset in memory as float32 tensors ($224 \times 224 \times 3$) requires $\approx 30.45\text{ GiB}$ of RAM. The production loader (`src/cropguard/data/loader.py`) enforces `cache_mode=None` (safe streaming) by default and supports disk-backed caching, raising an explicit warning if in-memory caching is requested for $>5,000$ samples.
5. **Upstream Pytest AST Deprecation Warnings (112 instances):**
   * Upstream TensorFlow 2.22 AutoGraph passes `___pyct_anno` to Python 3.14 AST constructors in `gast.astn.py:23`, generating benign deprecation warnings during test execution. All 32 tests pass and functional execution is unaffected.
6. **Preliminary Pipeline Throughput Clarification:**
   * The benchmark of **231.7 FPS** (EXP-007) measures raw input pipeline throughput (I/O reading, JPEG decode, resizing, normalization, and GPU prefetching on NVMe/CPU), not full CNN model training or inference latency.
7. **Unverified Botanical Relationships in Fallback Subset (24.30%):**
   * Full physical-leaf independence is strictly proven for the 75.70% of images indexed in author metadata (`leaf-map.json`). For the remaining 24.30% of fallback images (13,194 samples) and 11,834 single-image groups, grouping is based on normalized filename stems. While filename, exact MD5, and near-duplicate dHash leakage are completely eliminated, biological single-plant independence cannot be proven from author records due to legacy metadata omissions. Evaluation on `data/splits/classification_group_aware/` remains the most sound methodology achievable.
