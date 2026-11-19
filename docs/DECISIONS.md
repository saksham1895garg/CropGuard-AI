# CropGuard AI — Architectural & Technical Decisions Log

**Policy:** This document records all binding architectural, technical, and pedagogical decisions made for CropGuard AI. **No AI agent or collaborator may silently change, bypass, or revert an established decision.** Any proposed alteration must be justified and explicitly approved by the user.

---

## Decision Index

1. [ADR-001: Language Selection — Python over C++/Java](#adr-001)
2. [ADR-002: Primary Deep Learning Framework — TensorFlow / Keras](#adr-002)
3. [ADR-003: Dedicated Virtual Environment (.venv) Isolation](#adr-003)
4. [ADR-004: Interactive Jupyter Notebook-First Development Workflow](#adr-004)
5. [ADR-005: Object Detection Architecture — YOLO for Pest Localization](#adr-005)
6. [ADR-006: Dataset Selection & Ingestion Policy](#adr-006)
7. [ADR-007: Image Processing Stack — OpenCV and Pillow](#adr-007)
8. [ADR-008: Decision Support & Advisory Tool Positioning](#adr-008)
9. [ADR-009: Documentation-First Development & Handoff Standard](#adr-009)
10. [ADR-010: Multi-Modal Data Strategy, Stratified Partitioning & Imbalance Mitigation](#adr-010)
11. [ADR-011: Group-Aware Partitioning to Prevent Physical Leaf Leakage](#adr-011)
12. [ADR-012: Safe Configurable Memory Management in tf.data Ingestion](#adr-012)
13. [ADR-013: Model-Specific Input Preprocessing & Two-Phase Transfer Learning Protocol](#adr-013)
14. [ADR-014: Standalone Production CLI Training Pipeline & Strict Test Partition Isolation](#adr-014)
15. [ADR-015: U-Net Lesion Segmentation Architecture & Mask Binarization Protocol](#adr-015)
16. [ADR-016: Dual-Pipeline Production Streamlit Application & Academic Scope Boundary](#adr-016)

---

<a id="adr-001"></a>
### ADR-001: Language Selection — Python over C++/Java
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** While some generic University PBL rubrics mention C/C++/Java, the Project 05 course specifications and the Deep Learning lab plan (CSAI3027P) explicitly mandate Python, NumPy, Matplotlib, and modern deep learning frameworks.
- **Decision:** Python 3.14.4 is the exclusive programming language for CropGuard AI.
- **Rationale:**
  - Standard deep learning ecosystem (TensorFlow, Keras, PyTorch, YOLO, OpenCV) is natively supported in Python.
  - Interactive notebook prototyping and visualization are unmatched.
  - Academic clarification: Python requirement must be formally confirmed with faculty prior to final viva submission.

---

<a id="adr-002"></a>
### ADR-002: Primary Deep Learning Framework — TensorFlow / Keras
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** The course plan requires deep learning implementations with a modern framework.
- **Decision:** TensorFlow 2.22.0rc0 and Keras Nightly are selected as the primary framework for fundamental neural networks, CNNs, segmentation, autoencoders, and LSTM sequence models.
- **Rationale:**
  - Explicitly emphasized in the CSAI3027P syllabus.
  - Seamless GPU acceleration on NVIDIA RTX 5060 under WSL2 via pre-built CUDA 12 wheels.
  - PyTorch is permitted only for specialized sub-tasks (e.g., Ultralytics YOLO) if pre-built wheels are verified.

---

<a id="adr-003"></a>
### ADR-003: Dedicated Virtual Environment (.venv) Isolation
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** The user already had an existing working environment at `/home/saksham/tf-gpu` used by another project (SignBridge AI).
- **Decision:** Create and use an isolated `.venv` at `/home/saksham/CropGuard-AI/.venv`. Never install packages into or modify `/home/saksham/tf-gpu` or `/usr/bin/python3`.
- **Rationale:**
  - Prevents dependency conflicts between different coursework projects.
  - Ensures clean reproducibility when generating `requirements.txt`.
  - Enables dedicated kernel registration under the display name `"CropGuard AI (.venv)"`.

---

<a id="adr-004"></a>
### ADR-004: Interactive Jupyter Notebook-First Development Workflow
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** The user is a beginner in Deep Learning and learns by interactively running code cells, inspecting intermediate tensors, visualizing loss curves, and reviewing viva questions step-by-step.
- **Decision:** All theoretical concepts and algorithmic implementations must be introduced first through clean Jupyter notebooks (`.ipynb`) in `notebooks/`.
- **Core Rules:**
  1. AI agents generate clean, unexecuted notebooks (`outputs: []`, `execution_count: None`).
  2. AI agents **NEVER execute notebook cells headlessly** (`nbconvert`, `papermill`).
  3. The user manually executes cells in the interactive notebook editor.
  4. User execution confirmations are authoritative and completed notebook outputs must never be wiped or overwritten.
  5. Every notebook section adheres to the 7-part pedagogical structure: Explanation, Mathematical Formula, Practical Example, Code Cell, Expected Output, Interpretation, Viva Questions.

---

<a id="adr-005"></a>
### ADR-005: Object Detection Architecture — YOLO for Pest Localization
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** Crop surveillance requires both disease classification (whole leaf) and pest localization (small insects like aphids, whiteflies, mites).
- **Decision:** Adopt YOLO (You Only Look Once) as the primary architecture for agricultural pest detection.
- **Rationale:**
  - Real-time single-stage bounding-box regression suitable for edge deployment.
  - High accuracy on small object detection in cluttered foliage backgrounds.
  - Modular separation: Disease classification (CNN/Transfer Learning) is decoupled from Pest Detection (YOLO).

---

<a id="adr-006"></a>
### ADR-006: Dataset Selection & Ingestion Policy
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** Deep learning models require high-quality annotated agricultural imagery.
- **Decision:** Datasets will not be downloaded in initial fundamental phases. When ingestion begins:
  - **Disease Classification:** PlantVillage dataset (crowdsourced leaf images with healthy and diseased classes across tomato, potato, bell pepper, etc.).
  - **Pest Detection:** IP102 or CropPest dataset (benchmark pest bounding-box datasets).
- **Rules:**
  - Raw images are stored in `data/raw/` and strictly excluded from Git.
  - Every dataset must have a documented `DATASET_CARD.md` stating source, license, class balance, split strategy (Train/Val/Test), and known biases.

---

<a id="adr-007"></a>
### ADR-007: Image Processing Stack — OpenCV and Pillow
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** Images require color space conversion (RGB/BGR/HSV), resizing, normalization, and bounding-box drawing.
- **Decision:** OpenCV (`opencv-python-headless`) and Pillow (`PIL`) are designated as the official image processing libraries.
- **Rationale:** High-performance C-backends, industry standard, direct interoperability with NumPy arrays and TensorFlow tensors.

---

<a id="adr-008"></a>
### ADR-008: Decision Support & Advisory Tool Positioning
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** AI diagnosis in agriculture carries real-world risks if over-trusted.
- **Decision:** CropGuard AI is explicitly positioned as an educational and decision-support advisory system, not a definitive botanical diagnostic authority.
- **Rules:**
  - The UI must display model confidence scores, diagnostic limitations, and advisory disclaimers.
  - Never infer disease progression or temporal dynamics from a single static photograph.
  - Never fabricate test metrics or claim a model is trained without actual verifiable experiment logs.

---

<a id="adr-009"></a>
### ADR-009: Documentation-First Development & Handoff Standard
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** Multiple AI agents and sessions will collaborate on this codebase over time.
- **Decision:** Enforce a mandatory 7-stage development loop:
  $$\text{Read Documentation} \longrightarrow \text{Inspect Code} \longrightarrow \text{Understand Next Task} \longrightarrow \text{Implement} \longrightarrow \text{Test} \longrightarrow \text{Update Docs} \longrightarrow \text{Write Handoff Report}$$
- **Rules:**
  - No implementation may start without inspecting `docs/PROJECT_STATE.md` and `docs/NEXT_SESSION.md`.
  - Every task concludes with updating `docs/PROGRESS_LOG.md`, `docs/NEXT_SESSION.md`, and delivering the standard playbook handoff report.


---

<a id="adr-010"></a>
### ADR-010: Multi-Modal Data Strategy, Stratified Partitioning & Imbalance Mitigation
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-17
- **Context:** Ingestion of 54,305 PlantVillage images revealed severe class imbalance ($\rho = 36.23$, ranging from 5,507 down to 152 samples). Naive random splitting causes severe class distribution drift and test set sample starvation for rare classes. Additionally, multi-task learning requires standard coordinate conventions across segmentation and detection.
- **Decision:**
  1. Mandate deterministic, class-stratified 70/15/15 partitions generated via `scripts/generate_splits.py` with mathematical verification of zero sample leakage ($\text{Train} \cap \text{Val} = \emptyset, \dots$).
  2. Mandate inverse-frequency class loss weighting ($w_k = \frac{N_{\text{total}}}{K \cdot N_k}$) or Focal Loss in all classification model training loops to prevent majority class dominance.
  3. Standardize pest object detection annotations on normalized YOLO coordinates $[x_c, y_c, w, h] \in [0, 1]$.
  4. Implement data loading exclusively via TensorFlow `tf.data` with parallel JPEG decoding and `AUTOTUNE` prefetching.

---

<a id="adr-011"></a>
### ADR-011: Group-Aware Partitioning to Prevent Physical Leaf Leakage
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** An audit of 54,305 PlantVillage images against official author metadata (`leaf-map.json`) proved that naive 70/15/15 random splitting leaked 6,464 physical leaves (32.30% of all leaves in the dataset) across partitions. Furthermore, 21 duplicate MD5 hashes existed, allowing test images to share identical byte content or physical leaf structure with training images. This leads to memorization and invalidates generalization claims.
- **Decision:**
  1. Mandate group-aware leaf-level partitioning (`scripts/audit_plantvillage_leakage.py`) for all production training and evaluation.
  2. Partitions are stored in `data/splits/classification_group_aware/`: Train (35,221 / 64.86%), Val (8,085 / 14.89%), Test (10,999 / 20.25%).
  3. All images belonging to any physical leaf group must strictly reside within a single split ($\text{LeafGroups}(\text{Train}) \cap \text{LeafGroups}(\text{Test}) = \emptyset$).
  4. Exact duplicate MD5 hashes must never bridge training and evaluation sets ($\text{MD5}(\text{Train}) \cap \text{MD5}(\text{Test}) = \emptyset$).
  5. Preserve legacy naive splits in `data/splits/classification/` for historical reference and ablation auditing.

---

<a id="adr-012"></a>
### ADR-012: Safe Configurable Memory Management in tf.data Ingestion
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** In-memory caching (`dataset.cache()`) of 54,305 decoded float32 images at $224 \times 224 \times 3$ requires $\approx 30.45\text{ GiB}$ of RAM. Unconditionally caching the full dataset in RAM causes immediate host OOM kernel panic and training script aborts on standard developer workstations (16–32 GB RAM).
- **Decision:**
  1. Implement a safe, modular data loader in `src/cropguard/data/loader.py` with configurable `cache_mode`.
  2. Set `cache_mode=None` as the default and recommended mode, streaming samples dynamically from disk with AUTOTUNE prefetching, bounding RAM usage strictly to the active batch size.
  3. Support `cache_mode='disk'` to write cache files safely to `data/cache/`.
  4. Permit `cache_mode='memory'` only for small experiments, raising an explicit runtime warning if the dataset contains more than 5,000 samples.

---

<a id="adr-013"></a>
### ADR-013: Model-Specific Input Preprocessing & Two-Phase Transfer Learning Protocol
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** Comparative evaluation of Custom CNN, MobileNetV2, and ResNet50 revealed that applying identical preprocessing blindly corrupts deep feature representations. MobileNetV2 expects pixel values normalized to $[-1.0, 1.0]$ via $(x / 127.5) - 1.0$. ResNet50 expects raw $[0, 255]$ values converted from RGB to BGR with ImageNet channel means ($[103.939, 116.779, 123.68]$) subtracted. Feeding $[0, 1]$ scaled inputs into ResNet50 produces extreme negative activations ($\approx -103$), causing activation collapse.
- **Decision:**
  1. Mandate architecture-specific input preprocessing in `src/cropguard/classification/preprocessing.py` via `get_model_preprocessing(model_name)`.
  2. Integrate `preprocess_fn` dynamically into `build_classification_dataset` in `src/cropguard/data/loader.py`, allowing each model to consume correctly scaled tensors.
  3. Enforce a two-phase transfer learning workflow:
     - **Phase 1 (Feature Extraction):** Backbone weights are completely frozen (`trainable = False`), training only the custom 38-class classification head with $\eta = 10^{-3}$.
     - **Phase 2 (Fine-Tuning):** Top convolutional layers are progressively unfrozen while keeping Batch Normalization layers frozen, training end-to-end with low learning rate $\eta \le 10^{-5}$.
  4. Reserve the group-aware test partition exclusively for final evaluation; tune models and perform model selection using validation loss and validation Macro F1.

---

<a id="adr-014"></a>
### ADR-014: Standalone Production CLI Training Pipeline & Strict Test Partition Isolation
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** Interactive Jupyter notebooks are optimal for education, exploration, and quick pipeline smoke tests, but executing multi-epoch training across 35,221 images requires an interrupt-resilient, headless CLI tool. Furthermore, test partitions must be strictly held out to prevent data contamination during model tuning, and metrics such as ROC-AUC should be calculated without introducing heavyweight dependencies (e.g. Scikit-Learn or SciPy) into the minimal Python 3.14 environment.
- **Decision:**
  1. Author a dedicated standalone CLI training runner in `src/cropguard/classification/train.py`.
  2. Implement a two-phase transfer learning training loop (Phase 1: Feature Extraction, Phase 2: Fine-Tuning) with explicit protection of `BatchNormalization` layer statistics.
  3. Enforce strict isolation of `data/splits/classification_group_aware/test.csv` (10,999 images) during all training and validation workflows. The test set shall remain untouched until final benchmark reporting.
  4. Calculate One-vs-Rest Multiclass ROC-AUC via pure NumPy Wilcoxon-Mann-Whitney U rank sums ($O(K \cdot N \log N)$), resolving ties by average rank.
  5. Provide an instant `--dry-run` preflight mode validating arguments, partition counts, layer unfreezing, memory growth, and output directory scaffolding in $< 3$ seconds.
  6. Standardize production model artifacts under `models/classification/production/<model_name>/` containing `best_model.keras`, `training_history.csv`, `config.json`, `validation_metrics.json`, `confusion_matrix.png`, and `tensorboard/` event logs.

---

<a id="adr-015"></a>
### ADR-015: U-Net Lesion Segmentation Architecture & Mask Binarization Protocol
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** Foliar lesion segmentation requires pixel-level delineation of infected tissue across diverse crops and pathologies. PlantSeg ground-truth masks encode multi-class disease IDs $[1, 115]$ with 0 as healthy leaf/background. Furthermore, some smartphone captures contain EXIF orientation tags (tag 274: 6) that cause image and mask aspect ratio transposition if unhandled.
- **Decision:**
  1. Adopt a standard 4-stage U-Net encoder-decoder architecture with contracting blocks $(64, 128, 256, 512)$, bottleneck $(1024)$, expanding transposed convolutions $(512, 256, 128, 64)$ with skip connections, and a single-channel Sigmoid output layer (7,771,873 total parameters).
  2. Binarize all PlantSeg ground-truth masks strictly via $\text{mask} > 0 \to \{0.0, 1.0\}$, mapping all visible necrotic/chlorotic lesions into a single binary foreground target.
  3. Enforce spatial resizing of masks strictly via `Image.Resampling.NEAREST` to prevent interpolation blur and non-binary boundary values.
  4. Train using a balanced combined loss: $\mathcal{L} = 0.5 \cdot \text{BCE} + 0.5 \cdot \text{DiceLoss}$.
  5. Quarantined the official PlantSeg test partition (2,295 pairs) throughout training; evaluate exactly once on frozen weights with $\tau = 0.50$.

---

<a id="adr-016"></a>
### ADR-016: Dual-Pipeline Production Streamlit Application & Academic Scope Boundary
- **Status:** **Accepted & Enforced**
- **Date:** 2026-11-18
- **Context:** CropGuard AI is intentionally streamlined for academic presentation (UPES CSAI3027P) as a two-model diagnostic system: ResNet50 (classification) and U-Net (segmentation). Adding peripheral architectures (such as complex YOLO detectors, GANs, VAEs, or LSTMs) dilutes the primary core competencies and introduces unnecessary dependencies. The web UI must be beginner-friendly, reliable, and scientifically grounded.
- **Decision:**
  1. Standardize CropGuard AI as a unified two-model system combining frozen production ResNet50 (38 classes) and frozen production U-Net (foliar lesions) under `app/app.py`.
  2. Cache both model checkpoints in memory using `@st.cache_resource` to avoid multi-gigabyte reload overhead during interactive user sessions.
  3. Preprocess inputs using exact training specifications: ResNet50 receives $(224, 224, 3)$ with Caffe BGR mean subtraction; U-Net receives $(256, 256, 3)$ scaled to $[0.0, 1.0]$.
  4. Compute lesion severity percentage strictly from predicted binary mask pixels ($\ge \tau = 0.50$), never from classification confidence.
  5. Support full diagnostic export: JSON summary report and PNG lesion overlay.
  6. Prominently display the academic and agricultural disclaimer separating taxonomic classification from visible lesion segmentation and clarifying that visible lesions do not prove pathogen causation.
  7. Maintain check-ins and regression suite passing at 100% (61/61 tests). Checkpoints are permanently frozen.
