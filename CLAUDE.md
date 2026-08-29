# CLAUDE.md

## Project Overview

This repository contains a modular PyTorch workflow for training, evaluating, analyzing, and serving a Multilayer Perceptron (MLP) classifier on the MNIST dataset. It includes dynamic learning rate schedulers, error diagnostics, TensorBoard experiment tracking, and an interactive Streamlit web application with robust computer vision preprocessing.

## Local Workflow

- Install dependencies: `uv sync`
- Run unit tests: `uv run pytest -q`
- Lint and format: `uv run ruff format .` and `uv run ruff check . --fix`
- Train model: `uv run python scripts/train.py --epochs 20 --use-tensorboard`
- Launch TensorBoard: `uv run tensorboard --logdir artifacts/`
- Launch Streamlit web demo: `uv run streamlit run scripts/app.py`
- Training outputs are stored under `artifacts/<timestamp>/` and dataset files under `data/`

## Code Conventions

- Prefer small, explicit functions with Google-style docstrings (`Args`, `Returns`) and clear parameter names
- Use type annotations across public APIs, utilities, and dataclasses
- Add comments only where they clarify intent or non-obvious algorithms
- Keep unit tests fast, isolated, and focused on verifying model, preprocessing, and training components

## Repository Expectations

- Dataset files (`mnist_train.csv` and `mnist_test.csv`) are tracked via Git LFS
- Generated experiment run artifacts are stored in `artifacts/`
- Keep the test suite passing at 100% and linter rules clean before committing
