# MNIST Handwritten Digit Classifier (ANN)

A production-grade, modular PyTorch project for training, evaluating, analyzing, and interactively serving Multilayer Perceptron (MLP) neural networks on the MNIST dataset.

The project features dynamic learning rate schedulers, comprehensive error diagnostics (confusion matrices, misclassified sample galleries, classification reports), real-time experiment tracking with TensorBoard (gradient norms, layer histograms, HParams), and an interactive Streamlit web application with robust computer vision preprocessing for arbitrary-resolution images.

---

## 🌟 Key Features

- **End-to-End Data Pipeline**: CSV-based data loading with automatic train/val/test splitting, deterministic shuffling, and float32 pixel normalization.
- **Configurable MLP Architecture**: Modular `MLPClassifier` supporting arbitrary hidden layer depths, configurable dropout rates, and optional batch normalization.
- **Advanced Optimization & Schedulers**: Factory supporting `Adam`, `AdamW`, and `SGD` optimizers, paired with dynamic learning rate schedulers (`StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`).
- **Comprehensive Error Diagnostics**:
  - Training loss, validation loss, accuracy, and learning rate curves (`.png`).
  - Scikit-learn confusion matrix heatmaps.
  - Misclassified sample galleries showcasing the model's most confident classification errors.
  - Per-digit Precision, Recall, and F1-score serialization (`classification_report.json`).
  - Full experiment hyperparameter tracking (`training_args.json`).
- **Real-Time Experiment Tracking (TensorBoard)**:
  - Live scalar curves for training/validation loss and accuracy.
  - Layer-by-layer weight and bias distribution histograms.
  - Gradient $L_2$ norm tracking across individual layers and globally ($\|\nabla_\theta \mathcal{L}\|_2$) to detect vanishing/exploding gradients.
  - PyTorch computational graph visualization.
  - Hyperparameter tuning comparisons across runs via the HParams dashboard.
- **Interactive Streamlit Web Application**:
  - Live model checkpoint selection from `artifacts/`.
  - Real-time inference on arbitrary-resolution uploaded images or drawings.
  - Automatic color inversion for dark ink on white paper.
  - Confidence metrics, Top-3 candidate predictions, and interactive probability distribution bar charts.
- **Robust Computer Vision Preprocessing**:
  - **Otsu Global Binarization**: Automatically eliminates background paper noise and ambient shadows from real camera photos.
  - **Connected Component Analysis**: Filters out stray outlier dots and accidental pen specks.
  - **Adaptive Stroke Dilation**: Dynamically thickens 1-pixel hairline digital drawings to preserve stroke integrity through $24\times$ downsampling.
  - **NIST Center of Mass Alignment**: Centers digits according to their pixel mass centroid at $(14, 14)$, replicating the canonical MNIST alignment standard.

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
│   ├── app.py                     # Interactive Streamlit web application
│   └── train.py                   # Main CLI training & experiment entry point
├── src/
│   └── mnistdatasetann/
│       ├── args/
│       │   └── args_class.py      # Dataclass defining training hyperparameters
│       ├── data/
│       │   └── loader.py          # CSV loading, normalization, and splitting
│       ├── model/
│       │   ├── loader.py          # Robust checkpoint loading & weight restoration
│       │   ├── model.py           # Configurable PyTorch MLPClassifier architecture
│       │   └── trainer.py         # Training loop, scheduler stepping, & test evaluation
│       └── utils/
│           ├── diagnose.py        # Layer and global gradient norm computations
│           ├── preprocessor.py    # Robust Otsu & stroke-dilation image preprocessor
│           ├── printer.py         # Decorated console banner printer
│           ├── timer.py           # Context manager execution timer
│           └── visualizer.py      # Loss, accuracy, LR, confusion matrix, & error visualizers
├── tests/
│   └── test_mnist_dataset_ann.py  # Comprehensive unit test suite (14 test cases)
├── .gitattributes                 # Git LFS tracking configuration for large CSV files
├── .gitignore
├── CLAUDE.md                      # Developer environment guidelines
├── pyproject.toml                 # Package configuration, dependencies, and tools
├── README.md                      # Project documentation
└── uv.lock                        # Deterministic dependency lockfile
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Installation

Ensure you have **Python 3.12+** and [`uv`](https://github.com/astral-sh/uv) installed.

Clone the repository and synchronize all dependencies:

```bash
git clone https://github.com/Alok-Senapati/MnistDatasetANN.git
cd MnistDatasetANN
uv sync
```

*(Optional) If using standard pip:*
```bash
pip install -e .
```

---

### 2. Training the Model

Train an MLP classifier with learning rate scheduling and TensorBoard tracking:

```bash
uv run python scripts/train.py \
  --epochs 30 \
  --lr 1e-3 \
  --batch-size 256 \
  --hidden 256 128 64 \
  --use-batchnorm \
  --dropout 0.1 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --patience 15 \
  --use-tensorboard
```

#### Available CLI Arguments:
| Flag | Type | Default | Description |
|---|---|---|---|
| `--epochs` | `int` | `20` | Maximum training epochs |
| `--lr` | `float` | `1e-3` | Initial learning rate |
| `--batch-size` | `int` | `128` | Training mini-batch size |
| `--hidden` | `int ...` | `128 64` | Hidden layer dimensions |
| `--optimizer` | `str` | `adam` | Optimizer family (`adam`, `adamw`, `sgd`) |
| `--momentum` | `float` | `0.9` | Momentum coefficient for SGD |
| `--dropout` | `float` | `0.0` | Dropout probability after hidden layers |
| `--use-batchnorm` | `flag` | `False` | Enable batch normalization layers |
| `--weight-decay` | `float` | `0.0` | L2 weight regularization penalty |
| `--scheduler` | `str` | `none` | LR scheduler (`none`, `cosine`, `step`, `plateau`) |
| `--min-lr` | `float` | `1e-6` | Minimum learning rate floor |
| `--lr-decay-factor`| `float` | `0.5` | Decay factor $\gamma$ for StepLR / Plateau |
| `--lr-step-size` | `int` | `5` | Epoch interval for StepLR |
| `--patience` | `int` | `10` | Early stopping validation stall tolerance |
| `--use-tensorboard`| `bool` | `True` | Enable TensorBoard experiment logging |

---

### 3. Experiment Artifacts

Every training run creates a timestamped folder under `artifacts/<timestamp>/`:
- `best_model.pt`: Checkpoint containing architecture metadata, model weights, optimizer state, and best validation score.
- `accuracy_plot.png` & `loss_plot.png`: Epoch-by-epoch training vs validation curves.
- `lr_plot.png`: Optimizer learning rate progression.
- `confusion_matrix.png`: Normalized confusion matrix heatmap across digits 0–9.
- `misclassified_samples.png`: Visual gallery of top high-confidence test set mistakes.
- `classification_report.json`: Precision, recall, and F1-score breakdown per digit.
- `training_args.json`: Full hyperparameter configuration for exact reproducibility.
- `tensorboard/`: Event logs containing scalar curves, layer histograms, and HParams.

---

### 4. TensorBoard Experiment Dashboard

Launch TensorBoard to explore training runs, gradient norms, and hyperparameter comparisons:

```bash
uv run tensorboard --logdir artifacts/
```

Navigate to **`http://localhost:6006`** in your browser.

---

### 5. Interactive Streamlit Web Demo

Launch the interactive web UI to test the classifier on real-world photos or digital sketches:

```bash
uv run streamlit run scripts/app.py
```

- **Features**:
  - Model Run Selector (automatically finds all trained checkpoints).
  - Preprocessing Mode Selector:
    - **Otsu Mode**: Best for real camera photos, textured paper, and ambient shadows.
    - **Standard Mode**: Best for clean digital drawings (e.g., MS Paint).
  - Auto-Invert colors toggle.
  - Side-by-side visual comparison between the raw uploaded image and the normalized $28 \times 28$ input.
  - Real-time confidence metrics and interactive class probability charts.

---

## 🧪 Testing & Code Quality

Run the unit test suite:
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
