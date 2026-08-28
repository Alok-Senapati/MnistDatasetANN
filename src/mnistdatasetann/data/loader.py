"""MNIST dataset loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mnistdatasetann.utils import section_printer

DATA_PATH = Path(__file__).resolve().parents[3] / "data"


@dataclass(slots=True)
class MnistData:
    """Strongly typed container for the MNIST train/validation/test splits.

    Attributes:
        X_train: Feature matrix for the training split.
        y_train: Label vector for the training split.
        X_val: Feature matrix for the validation split.
        y_val: Label vector for the validation split.
        X_test: Feature matrix for the test split.
        y_test: Label vector for the test split.
        max_pixel_value: Maximum pixel value observed in the training split before scaling.
        is_normalized: Whether pixel values were normalized to the training max.
        label_proportions: Percentage distribution of digits in the training set.
        classes: Ordered list of digit classes present in the training data.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    max_pixel_value: float
    is_normalized: bool
    label_proportions: dict[str, float]
    classes: list[int]


@section_printer(section_name="MNIST Data Loaded")
def load_mnist_data(random_seed: int = 42, normalize: bool = True) -> MnistData:
    """Load the CSV-backed MNIST data set and split it into train/validation/test sets.

    Args:
        random_seed: Random state used for deterministic validation splitting.
        normalize: When True, divide all features by the maximum training value.

    Returns:
        A :class:`MnistData` object containing all splits and metadata.

    Raises:
        FileNotFoundError: If the expected training or test CSV files are missing.
    """
    train_path = DATA_PATH / "mnist_train.csv"
    test_path = DATA_PATH / "mnist_test.csv"

    if not train_path.exists() or not test_path.exists():
        msg = "MNIST CSV files were not found. Generate them with data/generate_mnist_csv.py."
        raise FileNotFoundError(msg)

    train = pd.read_csv(train_path, header=None)
    test = pd.read_csv(test_path, header=None)

    X = train.iloc[:, 1:].to_numpy()
    y = train.iloc[:, 0].to_numpy()
    X_test = test.iloc[:, 1:].to_numpy()
    y_test = test.iloc[:, 0].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        shuffle=True,
        random_state=random_seed,
    )

    max_value = float(X_train.max())

    if normalize:
        X_train = X_train / max_value
        X_val = X_val / max_value
        X_test = X_test / max_value

    label_values, counts = np.unique(y_train, return_counts=True)
    label_proportions = {
        str(int(label)): round(float(count * 100.0 / y_train.size), 2)
        for label, count in zip(label_values, counts, strict=False)
    }
    classes = [int(label) for label in label_values]

    print(
        "\n".join(
            [
                f"Train shape: {X_train.shape}",
                f"Train labels: {y_train.shape}",
                f"Validation shape: {X_val.shape}",
                f"Validation labels: {y_val.shape}",
                f"Test shape: {X_test.shape}",
                f"Test labels: {y_test.shape}",
                f"Label Proportions: {label_proportions}",
                f"Classes: {classes}",
            ]
        )
    )

    return MnistData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        max_pixel_value=max_value,
        is_normalized=normalize,
        label_proportions=label_proportions,
        classes=classes,
    )


def visualize_data() -> None:
    """Display a small grid of sample MNIST digits from the training split.

    Returns:
        None. The function renders a matplotlib figure to the screen.
    """
    dataset = load_mnist_data()
    X_train, y_train = dataset.X_train, dataset.y_train

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.ravel()

    for axis, label in zip(axes, range(10), strict=False):
        sample_index = np.where(y_train == label)[0][0]
        image = X_train[sample_index].reshape(28, 28)
        axis.imshow(image, cmap="gray")
        axis.set_title(f"Label {label}")
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_data()
