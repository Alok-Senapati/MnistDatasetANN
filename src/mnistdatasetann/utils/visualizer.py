"""Plotting helpers for training metrics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt


def _to_floats(values: Iterable) -> list[float]:
    """Convert a sequence of numeric values to a list of floats.

    Args:
        values: Iterable of numeric values to coerce to Python floats.

    Returns:
        A list of float values preserving the iterable order.
    """
    return [float(value) for value in values]


def visualize_loss(
    train_losses: Iterable,
    val_losses: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation loss over time.

    Args:
        train_losses: Per-epoch training-loss values.
        val_losses: Per-epoch validation-loss values.
        save_path: Optional output path for the PNG file.
    """
    train = _to_floats(train_losses)
    val = _to_floats(val_losses)
    epochs = list(range(1, len(train) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="Train Loss", color="tab:red")
    ax.plot(epochs, val, marker="x", linestyle="--", label="Val Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def visualize_accuracy(
    train_acc: Iterable,
    val_acc: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot training and validation accuracy over time.

    Args:
        train_acc: Per-epoch training accuracy values.
        val_acc: Per-epoch validation accuracy values.
        save_path: Optional output path for the PNG file.
    """
    train = _to_floats(train_acc)
    val = _to_floats(val_acc)
    epochs = list(range(1, len(train) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", label="Train Acc", color="tab:blue")
    ax.plot(epochs, val, marker="x", linestyle="--", label="Val Acc", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200)
        plt.close(fig)
    else:
        plt.show()
