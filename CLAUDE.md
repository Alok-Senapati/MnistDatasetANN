# CLAUDE.md

## Project overview

This repository contains a small PyTorch workflow for training a multilayer perceptron classifier on the MNIST dataset. It is intended to be easy to read, test, and extend for experiments.

## Local workflow

- Install dependencies with `uv sync`
- Run tests with `uv run pytest -q`
- Format and lint with `uv run ruff format .` and `uv run ruff check . --fix`
- Keep training outputs under `artifacts/` and generated dataset CSVs under `data/`

## Code conventions

- Prefer small, explicit functions with docstrings and clear parameter names
- Use type annotations for public APIs and data models
- Add comments only where they clarify intent or non-obvious logic
- Keep unit tests focused on behavior that is stable and independent of the full dataset pipeline

## Repository expectations

- The dataset should be generated before training if CSV files are missing
- Models and plots are saved under the `artifacts/` directory
- Do not commit generated model checkpoints or dataset files unless explicitly required
