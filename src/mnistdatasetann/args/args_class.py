"""Training configuration used by the CLI and training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class TrainingArgs:
    """Container for the hyperparameters and basic training configuration.

    The values are intentionally simple so they can be parsed from the command line
    and passed through the training pipeline without extra conversion logic.

    Attributes:
        epochs: Maximum number of training epochs to run.
        patience: Number of validation stalls allowed before early stopping triggers.
        dropout: Dropout probability applied after hidden layers.
        lr: Learning rate used by the optimizer.
        batch_size: Number of examples processed by each training mini-batch.
        optimizer: Optimizer family to use for gradient updates.
        momentum: Momentum coefficient for SGD optimization.
        hidden: Hidden-layer sizes of the MLP network.
        use_batchnorm: Whether to include batch normalization after hidden layers.
        weight_decay: L2 regularization strength applied by the optimizer.
    """

    epochs: int = 20
    patience: int = 10
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 128
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    momentum: float = 0.9
    hidden: list[int] = None  # type: ignore[assignment]
    use_batchnorm: bool = False
    weight_decay: float = 0.0
