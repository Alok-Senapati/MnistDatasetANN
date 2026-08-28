# MNIST Dataset ANN

A compact PyTorch project for training a small multilayer perceptron (MLP) on the MNIST handwritten-digit dataset. The project provides a CSV-based data loader, a reusable model definition, a training loop with checkpoints, and simple plotting utilities for loss and accuracy trends.

## Features

- Loads the MNIST dataset from CSV files in the `data/` folder
- Splits the training set into train/validation subsets
- Normalizes pixel values before model training
- Trains a compact MLP with configurable hidden layers, dropout, and batch normalization
- Saves the best checkpoint and plots for accuracy and loss
- Includes unit tests and Ruff-based formatting checks

## Project structure

```text
MnistDatasetANN/
├── data/
│   ├── generate_mnist_csv.py
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│   └── readme.md
├── scripts/
│   └── train.py
├── src/
│   └── mnistdatasetann/
│       ├── args/
│       ├── data/
│       ├── model/
│       └── utils/
├── tests/
├── .gitignore
├── CLAUDE.md
├── pyproject.toml
├── README.md
└── uv.lock
```

## Prerequisites

- Python 3.12+
- `uv` (recommended) or `pip`

## Installation

```bash
uv sync
```

If you are not using `uv`, install the package in editable mode:

```bash
python -m pip install -e .
```

## Preparing the dataset

The repository expects MNIST CSV files in `data/`.

If the files are missing, run:

```bash
python data/generate_mnist_csv.py
```

This script converts the IDX files into CSV format suitable for the project loader.

## Training a model

```bash
python scripts/train.py --epochs 20 --lr 1e-3 --batch-size 128 --hidden 128 64
```

The script stores artifacts such as checkpoints and plots under `artifacts/<timestamp>/`.

## Running tests

```bash
uv run pytest -q
```

## Linting and formatting

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Notes

- The project expects the dataset files to be present before training.
- Training and plotting utilities are intentionally lightweight and easy to extend.
- Checkpoints are saved only when validation loss improves.
