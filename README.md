# 🔢 MNIST Deep Learning, Computer Vision & Explainability System

A production-grade, modular PyTorch system for training, tuning, explaining, and interactively serving Deep Neural Networks (CNNs and MLPs) on the MNIST dataset.

The project features dynamic learning rate schedulers, high-performance GPU batch data augmentation, automated Optuna Bayesian hyperparameter search, Explainable AI (Vanilla Saliency & Grad-CAM), real-time TensorBoard experiment diagnostics, and an interactive Streamlit application with a live freehand drawing canvas and robust computer vision preprocessing.

> 📖 **Deep Dive Reference**: For comprehensive mathematical explanations, weight initialization theory, and architectural deep dives, see [**`LEARNINGS_AND_CONCEPTS.md`**](LEARNINGS_AND_CONCEPTS.md).

---

## 🌟 Key Features

- **Dual Architectures (CNN & MLP)**:
  - **`CNNClassifier`**: 2D Convolutions with Batch Normalization, Max Pooling, Dropout, and He/Kaiming initialization with spatial feature map extraction.
  - **`MLPClassifier`**: Fully connected multi-layer network with configurable hidden layer dimensions.
- **Explainable AI (XAI)**:
  - **Vanilla Saliency Maps**: First-order input pixel gradients ($\left| \frac{\partial S_c}{\partial X} \right|$) highlighting decisive stroke contours.
  - **Grad-CAM (Gradient-Weighted Class Activation Mapping)**: Hook-based channel activation weighting with Global Average Pooling ($\alpha_k^c$) and bilinear upsampled attention heatmaps.
- **High-Performance GPU Batch Augmentation**:
  - GPU-native stochastic affine (rotation, translation, scaling) and perspective transformations executing directly on CUDA tensors in parallel ($15\times$ faster than CPU pipelines).
- **Automated Hyperparameter Optimization (Optuna)**:
  - Bayesian Tree-structured Parzen Estimator (TPE) optimization with automated median trial pruning in `scripts/tune.py`.
- **Interactive Streamlit Web Application**:
  - **Live Freehand Drawing Canvas**: Customizable stroke width, dark/light canvas themes, and real-time stroke recognition.
  - **Image Upload & Preprocessing**: Robust Otsu binarization, connected component speckle filtering, adaptive stroke dilation, and NIST center-of-mass alignment.
  - **Live Explainability Dashboard**: Interactive Saliency and Grad-CAM attention heatmap overlays for any drawn digit.
- **Real-Time Experiment Tracking (TensorBoard)**:
  - Scalar curves for loss, accuracy, and learning rate.
  - Global gradient $L_2$ norm tracking ($\|\nabla_\theta \mathcal{L}\|_2$) across layers to catch vanishing/exploding gradients.
  - Multi-run parallel coordinate plots via the TensorBoard HParams dashboard.

---

## 📁 Project Structure

```text
MnistDatasetANN/
├── data/
│   ├── generate_mnist_csv.py      # Script to convert raw IDX files to CSV
│   ├── mnist_train.csv            # 60,000 training examples (tracked via Git LFS)
│   ├── mnist_test.csv             # 10,000 testing examples (tracked via Git LFS)
│   └── readme.md                  # Dataset formatting documentation
├── scripts/
│   ├── app.py                     # Streamlit web app with drawing canvas & XAI
│   ├── train.py                   # Main CLI training pipeline
│   └── tune.py                    # Automated Optuna hyperparameter optimization sweep
├── src/
│   └── mnistdatasetann/
│       ├── args/
│       │   └── args_class.py      # Dataclass defining training hyperparameters
│       ├── data/
│       │   ├── augment.py         # Stochastic transforms & dataset wrappers
│       │   └── loader.py          # CSV loading, normalization, and splitting
│       ├── explainability/
│       │   ├── gradcam.py         # Grad-CAM forward/backward hook engine
│       │   ├── saliency.py        # Input-gradient vanilla saliency generator
│       │   └── visualizer.py      # Multi-panel XAI heatmap overlay visualizer
│       ├── model/
│       │   ├── cnn.py             # Convolutional Neural Network architecture
│       │   ├── loader.py          # Robust checkpoint loading & weight restoration
│       │   ├── mlp.py             # Configurable PyTorch MLPClassifier architecture
│       │   └── trainer.py         # GPU-accelerated training loop & evaluation
│       └── utils/
│           ├── diagnose.py        # Layer and global gradient norm computations
│           ├── preprocessor.py    # Robust Otsu & stroke-dilation image preprocessor
│           ├── printer.py         # Decorated console banner printer
│           ├── timer.py           # Context manager execution timer
│           └── visualizer.py      # Loss, accuracy, LR, confusion matrix, & feature map visualizers
├── tests/
│   └── test_mnist_dataset_ann.py  # Comprehensive test suite (22 unit tests)
├── LEARNINGS_AND_CONCEPTS.md      # In-depth reference of deep learning & CV concepts
├── pyproject.toml                 # Package configuration, dependencies, and tools
└── README.md                      # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure you have **Python 3.12+** and [`uv`](https://github.com/astral-sh/uv) installed.

```bash
git clone https://github.com/Alok-Senapati/MnistDatasetANN.git
cd MnistDatasetANN
uv sync
```

---

### 2. Training the CNN Classifier

Train a CNN with GPU batch data augmentation, cosine annealing, and TensorBoard logging:

```bash
uv run python scripts/train.py \
  --model-type cnn \
  --use-augmentation \
  --augment-degrees 12.0 \
  --epochs 25 \
  --lr 1e-3 \
  --batch-size 256 \
  --conv-channels 32 64 \
  --fc-hidden 128 \
  --dropout 0.2 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --use-tensorboard
```

---

### 3. Automated Hyperparameter Tuning (Optuna)

Run a Bayesian optimization sweep across architectures, learning rates, and optimizers with trial pruning:

```bash
uv run python scripts/tune.py \
  --model-type cnn \
  --n-trials 20 \
  --epochs-per-trial 10 \
  --prune
```

---

### 4. Interactive Streamlit Web Application

Launch the web app to draw digits live, inspect convolutional activations, and generate Grad-CAM heatmaps:

```bash
uv run streamlit run scripts/app.py
```

---

## 🧪 Testing & Verification

Run the full automated test suite:
```bash
uv run pytest -q
```

Run linter and formatting checks:
```bash
uv run ruff check . --fix
uv run ruff format .
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
