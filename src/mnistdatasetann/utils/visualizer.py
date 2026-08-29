"""Plotting helpers for training metrics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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


def visualize_lr(
    lrs: Iterable,
    save_path: Path | None = None,
) -> None:
    """Plot learning rate decay over epochs.

    Args:
        lrs: Per-epoch learning rate values.
        save_path: Optional output path for the PNG file.
    """
    lr_values = _to_floats(lrs)
    epochs = list(range(1, len(lr_values) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, lr_values, marker="o", label="Learning Rate", color="tab:purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
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


def visualize_confusion_matrix(
    y_true: np.ndarray | Iterable,
    y_pred: np.ndarray | Iterable,
    classes: list[int] | list[str] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot the multi-class confusion matrix heatmap.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        classes: Ordered list of unique class labels.
        save_path: Optional output path for the PNG file.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_misclassified(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
    max_samples: int = 10,
    save_path: Path | None = None,
) -> None:
    """Plot a grid of sample images that were misclassified by the model.

    If probability scores are provided, the samples with highest prediction
    confidence on the wrong class (hard negatives) are displayed first.

    Args:
        images: 2D feature matrix of flattened image samples (N, 784).
        y_true: Ground truth labels.
        y_pred: Predicted class labels.
        y_probs: Optional predicted class probability distributions (N, num_classes).
        max_samples: Maximum number of misclassified samples to display.
        save_path: Optional output path for the PNG file.
    """
    mistake_indices = np.where(y_pred != y_true)[0]
    if len(mistake_indices) == 0:
        return

    if y_probs is not None:
        confidences = [y_probs[i, y_pred[i]] for i in mistake_indices]
        sorted_order = np.argsort(confidences)[::-1]
        selected_indices = mistake_indices[sorted_order[:max_samples]]
    else:
        selected_indices = mistake_indices[:max_samples]

    n_samples = len(selected_indices)
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for idx, sample_idx in enumerate(selected_indices):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img = images[sample_idx].reshape(28, 28)
        true_lbl = y_true[sample_idx]
        pred_lbl = y_pred[sample_idx]

        title = f"True: {true_lbl} | Pred: {pred_lbl}"
        if y_probs is not None:
            conf = y_probs[sample_idx, pred_lbl] * 100.0
            title += f"\nConf: {conf:.1f}%"

        ax.imshow(img, cmap="gray")
        ax.set_title(title, color="darkred", fontsize=10)
        ax.axis("off")

    for idx in range(n_samples, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
