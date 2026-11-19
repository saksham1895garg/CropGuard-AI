# CropGuard AI — Deep Learning Project Playbook

**Status:** Milestone 15 Completed — Final YOLO26n Held-Out Test Evaluation & Tri-Model Streamlit Integration  
**Frameworks:** TensorFlow 2.22 / Keras 3.x (Classification & Segmentation), PyTorch 2.15 / Ultralytics 8.4 (Pest Detection)  
**Primary Accelerator:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM)  
**Test Suite:** 77/77 passing regression tests across 9 modules  
**Web Application:** Streamlit live on `http://localhost:8501`  

---

## 1. Project Charter & Mission

CropGuard AI is an intelligent agricultural pathology and pest surveillance decision-support system designed to assist agriculturalists, students, and agronomists in identifying, segmenting, and detecting crop diseases and insect pests from leaf imagery.

### Core Model Portfolio:
1. **ResNet50 Classifier:** 38-class leaf pathology identification trained on group-aware PlantVillage partitions with strictly 0% leaf leakage (**98.42% held-out test accuracy** on 10,999 test samples). Checkpoint frozen.
2. **U-Net Segmenter:** Pixel-level foliar lesion segmentation trained on PlantSeg with Compound BCE + Dice loss (**63.48% held-out test Dice**, **46.49% test IoU**, **94.04% specificity** on 2,295 test pairs). Checkpoint frozen.
3. **YOLO26n Detector:** Agricultural insect pest detection powered by fine-tuned Ultralytics YOLO26n (2.58M parameters) across the 102-class IP102 benchmark taxonomy (**34.33% held-out test mAP@50**, **20.79% mAP@50-95**, **2.03 ms latency** on 3,798 test samples). Checkpoint frozen.

---

## 2. Milestone Completion Status

| Milestone | Scope / Subsystem | Artifacts & Evaluation Outcomes | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 01–06** | Mathematical & Deep Learning Foundations | Notebooks 01–06 covering neurons, backprop, tensor calculus, learning rules, CNN kernels, receptive fields, and agricultural vision exploratory data analysis. All cells interactively executed by user with figures preserved. | **Completed** |
| **Milestone 07** | Transfer Learning & CNN Architectures | Notebook 07 executed by user; comparative profiling of Custom CNN (429K params), MobileNetV2 (2.60M params), and ResNet50 (24.13M params). Inverse-frequency loss weighting ($w_k$) and Caffe preprocessing. | **Completed** |
| **Milestone 08** | Standalone Production Classification Pipeline | Standalone CLI training pipeline at `src/cropguard/classification/train.py`; 10,999-sample held-out test partition strictly quarantined. | **Completed** |
| **Milestone 09** | ResNet50 Final Evaluation | Final held-out test evaluation on 10,999 group-aware samples: **98.42% accuracy**, **0.9779 Macro F1**, **0.9998 ROC-AUC**; `reports/final_evaluation/` artifacts generated. Checkpoint frozen. | **Completed** |
| **Milestone 10** | Foliar Lesion Segmentation Subsystem | PlantSeg full census audit (9,163 train/val pairs); modular U-Net subsystem in `src/cropguard/segmentation/` (7.77M params); Compound BCE + Dice loss; Notebook 08 interactively executed by user. | **Completed** |
| **Milestone 11–12**| U-Net Production Training & Final Test Evaluation | Production training on 7,916 PlantSeg pairs (20 epochs, RTX 5060 GPU, **62.34% val Dice**); held-out test evaluation on 2,295 test pairs: **63.48% test Dice**, **46.49% test IoU**, **94.04% specificity**. Checkpoint frozen. | **Completed** |
| **Milestone 13** | YOLO Pest Detection & Streamlit Integration | IP102 local dataset audit; 102 benchmark classes ingested (`data/raw/ip102/classes.txt`); zero-fake-data policy strictly enforced; YOLO26n subsystem in `src/cropguard/detection/`; dual runtime verified (`pip check` clean); 8 tests in `tests/test_detection.py`; Streamlit app Section 4 ("Insect Pest Detection — YOLO") integrated with zero-detection safety cards. | **Completed** |
| **Milestone 14** | YOLO26n Training CLI, Stratified Split & GPU Smoke Test | Stratified 85/15 internal split (12,898 train, 2,280 val, 3,798 quarantined test); 100% converted labels (18,976); GPU smoke test completed in 20.84s (85.29% precision, 7.48% mAP@50, 4.13 ms latency); 12 tests in `tests/test_detection.py`; 74/74 tests passing repository-wide. | **Completed** |
| **Milestone 15** | YOLO26n Production Training Audit, Final Test Evaluation & Tri-Model App | Production training audit (20 epochs, val mAP@50 35.21%); held-out test evaluation on all 3,798 test images (**34.33% mAP@50**, **20.79% mAP@50-95**, **44.31% precision**, **38.64% recall**, **2.03 ms latency**); 0% leakage verified; full tri-model Streamlit dashboard live with 5 presets; **77/77 passing regression tests**. All checkpoints frozen. | **Completed** |

---

## 3. Technology Stack & Dual Runtime

- **Language:** Python 3.14.4 (WSL2 Linux x86_64)
- **Virtual Environment:** Dedicated `.venv` in workspace (`/home/saksham/CropGuard-AI/.venv`)
- **Classification & Segmentation:** TensorFlow 2.22.0rc0, Keras Nightly (`3.16.0.dev2026110903`) with dynamic GPU memory growth
- **Pest Detection:** PyTorch 2.15.0+cu132, Torchvision 0.29.2, Ultralytics 8.4.215 with native CUDA acceleration
- **Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB VRAM, Driver: 596.21, Host CUDA: 13.2)
- **Web Interface:** Streamlit 1.67.0 running on `http://localhost:8501`
- **Testing:** pytest 9.2.0 (**77/77 passing regression tests**)

---

## 4. Operational Guidelines & Anti-Fabrication Standards

1. **Interactive Notebook-First Rule:** Educational notebooks (`01` through `08`) belong to the student. Completed notebooks have user-executed outputs and figures that must never be wiped or overwritten.
2. **Strict Dataset & Detection Honesty:** If a dataset (such as IP102) is not downloaded locally due to upstream access blocks (e.g. Google Drive auth), do not fabricate synthetic data or simulate fake detectors. Explicitly report the audit findings, provide manual download instructions, and report `0 insect pests detected` on clean foliar images without hallucinating bounding boxes.
3. **Decoupled Architecture:** Maintain clean separation between whole-leaf classification (ResNet50), pixel lesion segmentation (U-Net), and object detection (YOLO26n).
4. **Zero-Leakage Guarantee:** Never evaluate on training or validation splits. Held-out test splits (`data/splits/classification_group_aware/test.csv` and `data/splits/segmentation/test.csv`) are evaluated exactly once and permanently frozen.
5. **Git Protection:** Do not commit or push to remote repositories without explicit user instruction.
