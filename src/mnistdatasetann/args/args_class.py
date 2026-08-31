"""Training configuration used by the CLI and training loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class TrainingArgs:
    """Container for the hyperparameters and basic training configuration.

    The values are intentionally simple so they can be parsed from the command line
    and passed through the training pipeline without extra conversion logic.

    Attributes:
        model_type: Architecture to instantiate (`mlp` or `cnn`).
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
        scheduler: Scheduler family to use for dynamic learning rate updates.
        min_lr: Minimum learning rate floor for schedulers.
        lr_decay_factor: Multiplicative factor for plateau and step LR decay.
        lr_step_size: Epoch interval for StepLR decay.
        use_tensorboard: Whether to enable TensorBoard for tracking and diagnostics.
        conv_channels: List of output channel depths for each convolutional block.
        fc_hidden: Number of hidden units in the dense classification head.
        use_augmentation: Whether to enable stochastic on-the-fly data augmentation.
        augment_degrees: Maximum random rotation angle in degrees `(-degrees, +degrees)`.
        augment_translate: Maximum random translation shift fraction.
        augment_scale_min: Minimum random scale factor.
        augment_scale_max: Maximum random scale factor.
    """

    model_type: Literal["mlp", "cnn"] = "mlp"
    epochs: int = 20
    patience: int = 10
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 128
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    momentum: float = 0.9
    hidden: list[int] = field(default_factory=lambda: [128, 64])
    use_batchnorm: bool = False
    weight_decay: float = 0.0
    scheduler: Literal["none", "cosine", "plateau", "step"] = "none"
    min_lr: float = 1e-6
    lr_decay_factor: float = 0.5
    lr_step_size: int = 5
    use_tensorboard: bool = True
    conv_channels: list[int] = field(default_factory=lambda: [32, 64])
    fc_hidden: int = 128
    use_augmentation: bool = False
    augment_degrees: float = 12.0
    augment_translate: float = 0.08
    augment_scale_min: float = 0.92
    augment_scale_max: float = 1.08
