"""Training utilities and optimizer factory for the MNIST classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from mnistdatasetann.utils import (
    Timer,
    section_printer,
    visualize_accuracy,
    visualize_loss,
    visualize_lr,
)


def get_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> Optimizer:
    """Create the requested optimizer instance for the supplied model.

    Args:
        model: The PyTorch model whose parameters will be optimized.
        optimizer: One of ``adam``, ``adamw``, or ``sgd``.
        lr: Learning rate used by the optimizer.
        weight_decay: L2 penalty coefficient.
        momentum: Momentum coefficient for SGD.

    Returns:
        The matching PyTorch optimizer.

    Raises:
        ValueError: If the optimizer name is unsupported.
    """
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )

    valid = ["adam", "adamw", "sgd"]
    raise ValueError(f"Invalid optimizer. Please select from {valid}.")


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: Literal["none", "step", "cosine", "plateau"],
    min_lr: float,
    lr_decay_factor: float,
    lr_step_size: int,
) -> LRScheduler | None:
    """
    Create the requested LRScheduler for the supplied optimizer.

    Args:
        optimizer: Optimizer whose lr will be updated by the scheduler
        scheduler_name: Name of the scheduler to use
        min_lr: Minimum learning rate
        lr_decay_factor: Factor γ for plateau/step decay
        lr_step_size: Epoch interval for StepLR

    Returns:
        Requested LRScheduler.
    """
    match scheduler_name:
        case "step":
            return StepLR(optimizer=optimizer, step_size=lr_step_size, gamma=lr_decay_factor)
        case "cosine":
            return CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=min_lr)
        case "plateau":
            return ReduceLROnPlateau(
                optimizer=optimizer, mode="min", factor=lr_decay_factor, patience=3, min_lr=min_lr
            )
        case _:
            return None


@section_printer("Model Training")
def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    epochs: int,
    model_init_args: dict,
    device: Literal["cpu", "cuda"],
    lr_scheduler: LRScheduler | None,
    output_path: Path,
    patience: int = 10,
) -> nn.Module:
    """Train the model with early stopping and save the best checkpoint.

    Args:
        train_loader: Data loader that yields the training batches.
        val_loader: Data loader used to evaluate the validation set each epoch.
        model: Model instance to optimize and validate.
        criterion: Loss function used for training and validation scoring.
        optimizer: Optimizer responsible for parameter updates.
        epochs: Maximum number of epochs to run before stopping.
        model_init_args: Initial configuration used to describe the model architecture.
        device: Device on which tensor operations should execute.
        output_path: Directory used for checkpoints and training-curve plots.
        patience: Number of epochs without validation-improvement before early stopping.

    Returns:
        The trained model instance after the final epoch.
    """
    training_losses: list[float] = []
    training_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    best_val_loss = float("inf")
    degrade_counter = 0
    learning_rates: list[float] = []

    for epoch in range(1, epochs + 1):
        with Timer() as elapsed_timer:
            model.train()
            running_training_loss = 0.0
            running_training_correct = 0.0
            train_seen = 0

            for xb, yb in train_loader:
                optimizer.zero_grad()
                xb, yb = xb.to(device), yb.to(device)
                logits_train = model(xb)
                loss = criterion(logits_train, yb)
                loss.backward()
                optimizer.step()

                running_training_loss += loss.item() * xb.size(0)
                predicted_train = logits_train.argmax(dim=1)
                running_training_correct += (predicted_train == yb).sum().item()
                train_seen += xb.size(0)

            train_loss = float(running_training_loss / train_seen)
            train_accuracy = float(running_training_correct / train_seen)
            training_losses.append(train_loss)
            training_accuracies.append(train_accuracy)

            model.eval()
            running_val_loss = 0.0
            running_val_correct = 0.0
            val_seen = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits_val = model(xb)
                    loss = criterion(logits_val, yb)
                    running_val_loss += loss.item() * xb.size(0)
                    predicted_val = logits_val.argmax(dim=1)
                    running_val_correct += (predicted_val == yb).sum().item()
                    val_seen += xb.size(0)

            val_loss = float(running_val_loss / val_seen)
            val_accuracy = float(running_val_correct / val_seen)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if val_loss < best_val_loss:
                degrade_counter = 0
                best_val_loss = val_loss
                checkpoint = {
                    "model_meta": {
                        "class_name": model.__class__.__name__,
                        "module_path": model.__class__.__module__,
                        "init_args": model_init_args,
                        "arch_str": str(model),
                    },
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "val_acc": val_accuracy,
                }
                torch.save(checkpoint, output_path / "best_model.pt")
            else:
                degrade_counter += 1

            if degrade_counter >= patience:
                print(f"Early stopping at epoch: {epoch}...")
                break

        learning_rates.append(optimizer.param_groups[0]["lr"])
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

        print(
            f"Epoch: {epoch:02d}/{epochs}, "
            f"Time: {elapsed_timer.elapsed:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f} "
            f"| Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f} "
            f"| Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f} "
        )

    visualize_accuracy(
        training_accuracies, val_accuracies, save_path=output_path / "accuracy_plot.png"
    )
    visualize_loss(training_losses, val_losses, save_path=output_path / "loss_plot.png")
    visualize_lr(learning_rates, save_path=output_path / "lr_plot.png")
    return model
