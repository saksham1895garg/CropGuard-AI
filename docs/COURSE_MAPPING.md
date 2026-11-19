# CropGuard AI — Academic Course Mapping (CSAI3027P)

**Course:** Deep Learning (CSAI3027P)  
**Academic Institution:** UPES, School of Computer Science (2026–27)  
**Curriculum Alignment:** University Project-Based Learning (PBL) & Deep Learning Lab Curriculum  

---

## 1. Course Outcomes (CO) Overview

| Course Outcome | Description | Direct Project Implementation |
| :--- | :--- | :--- |
| **CO1** | Understand and formulate the mathematical, biological, and structural foundations of artificial neural networks, activation functions, perceptrons, feed-forward architectures, backpropagation calculus, optimization algorithms (GD, SGD, Momentum, Nesterov), and Singular Value Decomposition (SVD). | Notebook 01 (Neurons & MLP), Notebook 02 (Backprop & Optimizers), Notebook 03 (SVD, Leaf Image Compression) |
| **CO2** | Apply multidimensional tensor computation, array broadcasting rules, inner/dot products, dimensional transformations (reshaping, transposing, flattening), and automatic differentiation (`tf.GradientTape`) to construct deep learning workflows. | Notebook 03 (Tensor Operations, Broadcasting, Auto-Diff & Hardware Acceleration) |
| **CO3** | Analyze and implement fundamental learning rules (Hebbian, Modified Hebbian/Oja's, Competitive, Error-Correction, Boltzmann), pattern association, pattern recognition, function approximation, and generalization/overfitting strategies. | Notebook 04 (Learning Rules & Neural Network Training Dynamics) |
| **CO4** | Design, train, evaluate, and benchmark deep convolutional neural networks, semantic segmentation pipelines, and object detectors (YOLO) for real-world agricultural disease classification and pest localization. | Notebook 05 (CNN Feature Maps & Pooling), Notebook 06 (Agricultural Vision Exploration & tf.data), `src/cropguard/classification`, `src/cropguard/detection` |
| **CO5** | Explore advanced generative architectures (VAE/GAN) and recurrent sequence models (LSTM) for disease severity progression forecasting and synthetic symptom augmentation. | `src/cropguard/sequence` (LSTM surveillance), `src/cropguard/generative` (Leaf symptom synthesis) |

---

## 2. Unit-by-Unit Syllabus & Artifact Mapping

### Unit I: Introduction to Deep Learning, Feed-Forward Networks, Optimizers & SVD
* **Official Syllabus Topics:** Historical evolution, McCulloch-Pitts model, Perceptrons, Sigmoid neurons, Multi-Layer Perceptron (MLP), feed-forward networks, backpropagation, Gradient Descent (GD), Momentum, Nesterov Accelerated Gradient (NAG), Stochastic Gradient Descent (SGD), Singular Value Decomposition (SVD), parameters vs. hyperparameters.
* **Lab Topics:** Python & environment setup, manual perceptron logic, building forward and backward passes from scratch in NumPy, optimizer trajectories and convergence comparisons, low-rank SVD matrix approximations.
* **Project Artifacts:**
  - [`notebooks/01_neurons_and_mlp.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/01_neurons_and_mlp.ipynb): **[Completed & User Executed]** McCulloch-Pitts thresholding, Sigmoid non-linear mapping, Perceptron linear decision boundaries, XOR non-linear separability problem, Multilayer Perceptron forward pass.
  - [`notebooks/02_backpropagation_and_optimizers.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/02_backpropagation_and_optimizers.ipynb): **[Completed & User Executed]** Forward propagation, loss function geometry (MSE vs. BCE), analytical chain rule derivation ($\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \sigma'(z) \cdot x$), step-by-step numerical backpropagation, 2D saddle-point convergence comparison of GD, SGD, Momentum, NAG, and Adam.
  - [`notebooks/03_tensor_operations_broadcasting_and_svd.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/03_tensor_operations_broadcasting_and_svd.ipynb) *(Topics 10–12)*: **[Completed & User Executed]** SVD mathematical decomposition ($\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$), Eckart-Young-Mirsky Theorem, Scree plots, cumulative energy curves, procedural leaf image compression across ranks $k \in \{2, 8, 25, 50\}$.

---

### Unit II: Tensor Computation, Broadcasting & Gradient Calculation
* **Official Syllabus Topics:** Tensor operations, array broadcasting, inner products, tensor reshaping, transposing, flattening, and gradient calculation using automatic differentiation.
* **Lab Topics:** Tensor ranks and shapes in NumPy and TensorFlow, implementing broadcasting operations, cosine similarity and vector dot products, multi-dimensional array transformations, evaluating derivatives via `tf.GradientTape`, CPU vs. GPU hardware benchmarking.
* **Project Artifacts:**
  - [`notebooks/03_tensor_operations_broadcasting_and_svd.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/03_tensor_operations_broadcasting_and_svd.ipynb) *(Topics 1–9, 13)*: **[Completed & User Executed]** Tensor dimensional hierarchy ($0\text{D} \to 4\text{D}$), slicing and ROI extraction, element-wise vs matrix operations, formal 2-rule array broadcasting, cosine similarity matching of leaf symptoms, batch matrix multiplication, `tf.GradientTape` higher-order auto-differentiation, CPU vs. RTX 5060 GPU benchmark.

---

### Unit III: Learning Rules & Neural Network Training Dynamics
* **Theory Topics:** Hebbian learning, modified Hebbian learning (Oja's rule), competitive learning (Winner-Take-All), error-correction learning (Delta/Widrow-Hoff rule), Boltzmann learning, pattern association, pattern recognition, function approximation, generalization, overfitting, regularization ($L_1, L_2$ weight decay), dropout, batch normalization.
* **Lab Topics:** Implementing classical learning rules, observing stability/instability, training MLPs for non-linear function approximation, diagnosing overfitting via training vs. validation curves.
* **Project Artifacts:**
  - [`notebooks/04_learning_rules_and_training.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/04_learning_rules_and_training.ipynb): **[Completed & User Executed]**
    - Topic 1: Hebbian Learning ($\Delta w_i = \eta y x_i$) & unbounded weight explosion proof.
    - Topic 2: Modified Hebbian / Oja's Rule ($\Delta w = \eta (y x - y^2 w)$) & PCA unit-norm alignment.
    - Topic 3: Competitive Learning & prototype clustering of leaf disease symptoms.
    - Topic 4: Error-Correction / Delta Rule on smooth quadratic error surfaces.
    - Topic 5: Boltzmann Learning, simulated annealing, energy states & stochastic neurons.
    - Topic 6: Pattern Association (Hopfield auto-associative memory with 30% noise tolerance).
    - Topic 7: Pattern Recognition & multi-class decision regions for plant pathology.
    - Topic 8: Function Approximation of non-linear disease spread dynamics.
    - Topic 9: Generalization, Overfitting, $L_2$ Weight Decay & Dropout in TensorFlow/Keras.

---

### Unit IV: Convolutional Neural Networks, Classification & Object Detection
* **Theory Topics:** 2D and 3D convolutions, convolution arithmetic (padding, stride, dilation), feature maps, pooling layers (Max, Average, Global Average), receptive field growth, dense transition parameter calculus, classic CNN architectures (LeNet, AlexNet, VGG, ResNet, MobileNet), Object Detection principles, one-stage detectors (YOLO), Intersection over Union (IoU), Mean Average Precision (mAP).
* **Lab Topics:** Visualizing convolutional filters and feature map activations, building a CNN classifier for PlantVillage leaf diseases, fine-tuning pretrained backbones via Transfer Learning, deploying a YOLO pest detector.
* **Project Artifacts:**
  - [`notebooks/05_cnn_feature_maps_and_pooling.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/05_cnn_feature_maps_and_pooling.ipynb): **[Completed & User Executed]**
    - 2D spatial cross-correlation & discrete convolution.
    - Manual Sobel horizontal/vertical edge filters on leaf structures.
    - Universal spatial dimension formula: $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$.
    - Stride downsampling, VALID vs SAME padding, ReLU sparsity.
    - Max Pooling vs Average Pooling vs Global Average Pooling.
    - 3D multi-channel convolutions ($K_h \times K_w \times C_{in} \times C_{out}$).
    - Receptive Field linear growth calculus: $RF_{l} = RF_{l-1} + (k - 1) \cdot j_{l-1}$.
    - Parameter calculus of dense transition: $N_{params} = (H_{out} \cdot W_{out} \cdot C_{out} + 1) \cdot N_{dense}$.
    - Keras functional foliar CNN & intermediate feature map extraction on RTX 5060 GPU.
  - [`notebooks/06_agricultural_dataset_exploration.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/06_agricultural_dataset_exploration.ipynb): **[Completed & User Executed]**
    - Agricultural computer vision spectrum (Classification vs Lesion Segmentation vs Pest Detection).
    - PlantVillage dataset audit (54,305 color images, 38 classes, 0 corrupt, uniform $256 \times 256$).
    - Chlorophyll degradation RGB histograms (Healthy green dominance vs necrotic brown shifts).
    - Severe class imbalance calculus ($\rho = 36.80$) & inverse-frequency loss weights ($w_k$).
    - Solanaceae and staple crop multi-class leaf gallery.
    - Mathematical proof of stratified leak-free splitting ($\text{Train} \cap \text{Val} = \emptyset$).
    - Foliar lesion semantic segmentation overlay (PlantSeg) & severity coverage ratio.
    - Agricultural pest bounding boxes (IP102) & PASCAL VOC to normalized YOLO conversion.
    - End-to-end production `tf.data` GPU-prefetching pipeline (**231.7 FPS input pipeline throughput**).
  - [`notebooks/07_cnn_architectures_and_transfer_learning.ipynb`](file:///home/saksham/CropGuard-AI/notebooks/07_cnn_architectures_and_transfer_learning.ipynb): **[Created & Verified — Ready for User Execution]**
    - Comparative foliar disease classification across 38 categories using group-aware partitions.
    - Custom 4-stage Foliar CNN baseline (~350k parameters) with Conv2D-BatchNorm-ReLU-MaxPool.
    - Transfer Learning with MobileNetV2 (~2.3M params): Depthwise separable convolutions & inverted residuals ($1/N + 1/D_k^2 \approx 1/9\times$ computation).
    - Transfer Learning with ResNet50 (~24M params): Residual skip connections ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$) overcoming vanishing gradients.
    - Exact model-specific preprocessing pipelines (Custom: $[0, 1]$, MobileNetV2: $[-1, 1]$, ResNet50: BGR mean subtraction $[B-103.939, G-116.779, R-123.68]$).
    - Mathematically derived inverse-frequency class weights ($w_k = \frac{N}{K \cdot N_k}$) balancing the $36.8\times$ sample disparity.
    - Pure NumPy multi-class evaluation metrics (Accuracy, Macro Precision, Macro Recall, Macro F1, $38\times 38$ Confusion Matrix).
    - Empirical hardware inference latency profiling (ms/image) on RTX 5060 GPU with warm-up.
    - Diagnostic error analysis, foliar visual confusion, and critical discussion of agricultural confounding / domain shift.
    - 3 Hands-on Exercises (Focal Loss, Grad-CAM, Knowledge Distillation) and 10 in-depth Viva Voce Questions & Answers.
  - Classification Subsystem: `src/cropguard/classification/` (`models.py`, `preprocessing.py`, `training.py`, `evaluation.py`).
  - Data Pipeline Scripts: `scripts/prepare_plantvillage.py`, `scripts/audit_plantvillage_leakage.py`, `scripts/prepare_plantseg.py`, `scripts/prepare_ip102.py`, `scripts/generate_splits.py`, `src/cropguard/data/loader.py`, `scripts/inspect_datasets.py`, `scripts/visualize_samples.py`.
  - Regression Test Suite: `tests/` (**22/22 passing tests** across environment, data pipeline, and classification).

---

### Unit V: Recurrent Models, Generative Architectures & Decision Support Interface
* **Theory Topics:** Recurrent Neural Networks (RNN), vanishing/exploding gradients, Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), Autoencoders, Variational Autoencoders (VAE), Generative Adversarial Networks (GAN).
* **Lab Topics:** Time-series forecasting with LSTM, image reconstruction with Autoencoders, synthetic sample generation with GANs, building an interactive application.
* **Project Artifacts (Planned):**
  - `src/cropguard/sequence/`: LSTM model for temporal disease progression forecasting based on humidity, temperature, and historical severity scores.
  - `src/cropguard/generative/`: Conditional GAN / VAE for synthesizing minority class disease leaf samples to alleviate class imbalance.
  - `app/app.py`: Streamlit surveillance dashboard presenting real-time image upload, disease classification, pest bounding boxes, Grad-CAM interpretability overlays, and severity reports.

---

## 3. Evaluation & Academic Viva Defense Preparedness

Every notebook and module includes specific Viva Q&A sections designed to prepare the student for academic examination:
1. **Mathematical Derivations:** Ability to derive the Delta rule, backpropagation weight updates, SVD outer product expansion, universal convolution dimension formula, and VOC-to-YOLO normalized coordinates.
2. **Architectural Justifications:** Explaining why YOLO is chosen over Faster R-CNN for edge pest surveillance (single-pass real-time inference vs. two-stage region proposal latency).
3. **Failure Mode Analysis:** Demonstrating why Hebbian learning causes unbounded weight growth without Oja's normalization, why vanilla Gradient Descent oscillates in high-curvature ravines, and why accuracy fails under $36.23\times$ class imbalance.
4. **Hardware Understanding:** Explaining how CUDA tensors, memory growth, PTX JIT compilation, and `tf.data.AUTOTUNE` prefetching function on NVIDIA Blackwell laptop hardware.
