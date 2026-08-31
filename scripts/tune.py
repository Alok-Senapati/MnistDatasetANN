"""Automated hyperparameter optimization pipeline using Optuna and Bayesian search."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

from mnistdatasetann.data import get_train_transforms, load_mnist_data
from mnistdatasetann.model import (
    CNNClassifier,
    MLPClassifier,
    get_optimizer,
    get_scheduler,
)
from mnistdatasetann.utils import section_printer

# Suppress verbose Optuna logging during evaluation
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 42
BASE_ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "artifacts"


def parse_tune_args() -> argparse.Namespace:
    """Parse command line arguments for the hyperparameter tuning sweep.

    Returns:
        Populated namespace containing tuning search settings.
    """
    parser = argparse.ArgumentParser(
        description="MNIST Hyperparameter Optimization (Optuna).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "mlp", "both"],
        default="cnn",
        help="Architecture family to optimize.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna optimization trials to execute.",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=8,
        help="Maximum number of epochs to train each candidate trial.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mnist_tuning_study",
        help="Name identifier for the Optuna study.",
    )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automated trial early stopping for underperforming candidates.",
    )
    return parser.parse_args()


def objective(
    trial: optuna.Trial,
    dataset_tensors: dict[str, torch.Tensor],
    image_shape: tuple[int, int, int],
    classes: list[int] | list[str],
    model_type_choice: str,
    epochs: int,
    device: str,
    enable_pruner: bool,
) -> float:
    """Objective function executed for each candidate hyperparameter trial.

    Args:
        trial: Current Optuna trial instance.
        dataset_tensors: Pre-loaded training, validation, and test PyTorch tensors.
        image_shape: Spatial tuple `(in_channels, height, width)`.
        classes: List of unique class labels.
        model_type_choice: Chosen model architecture (`'cnn'`, `'mlp'`, or `'both'`).
        epochs: Number of training epochs per candidate.
        device: Target execution device (`'cuda'` or `'cpu'`).
        enable_pruner: Whether to report intermediate metrics for early pruning.

    Returns:
        Best validation accuracy achieved by this trial candidate.
    """
    # 1. Sample Model Family & Architecture Hyperparameters
    if model_type_choice == "both":
        model_type = trial.suggest_categorical("model_type", ["cnn", "mlp"])
    else:
        model_type = model_type_choice

    # 2. Sample Training & Regularization Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step", "none"])

    # 3. Sample Augmentation Settings
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    gpu_transforms = None
    if use_augmentation:
        degrees = trial.suggest_float("augment_degrees", 5.0, 15.0, step=5.0)
        translate = trial.suggest_float("augment_translate", 0.04, 0.10, step=0.02)
        gpu_transforms = get_train_transforms(
            degrees=degrees, translate=translate, scale=(0.92, 1.08)
        ).to(device)

    # 4. Instantiate Model Architecture
    n_classes = len(classes)
    if model_type == "cnn":
        conv_config_name = trial.suggest_categorical(
            "conv_config", ["small_16_32", "medium_32_64", "large_32_64_128"]
        )
        channel_map = {
            "small_16_32": [16, 32],
            "medium_32_64": [32, 64],
            "large_32_64_128": [32, 64, 128],
        }
        conv_channels = channel_map[conv_config_name]
        fc_hidden = trial.suggest_categorical("fc_hidden", [64, 128, 256])

        model: nn.Module = CNNClassifier(
            in_dims=image_shape,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            num_classes=n_classes,
            dropout=dropout,
        ).to(device)
    else:
        hidden_config_name = trial.suggest_categorical(
            "hidden_config", ["128_64", "256_128", "256_128_64"]
        )
        hidden_map = {
            "128_64": [128, 64],
            "256_128": [256, 128],
            "256_128_64": [256, 128, 64],
        }
        hidden = hidden_map[hidden_config_name]
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])

        model = MLPClassifier(
            in_dim=784,
            out_dim=n_classes,
            hidden=hidden,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        ).to(device)

    # 5. Setup Data Loaders
    train_dataset = TensorDataset(dataset_tensors["X_train"], dataset_tensors["y_train"])
    val_dataset = TensorDataset(dataset_tensors["X_val"], dataset_tensors["y_val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name, lr, weight_decay, momentum=0.9)
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        min_lr=1e-6,
        lr_decay_factor=0.5,
        lr_step_size=4,
    )

    best_val_acc = 0.0

    # 6. Mini Training Loop with Optuna Pruner Reporting
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if gpu_transforms is not None:
                xb = xb.reshape(-1, 1, 28, 28)
                xb = gpu_transforms(xb)

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation Step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits_val = model(xb)
                pred = logits_val.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += xb.size(0)

        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # 7. Report intermediate metric for automated pruning
        if enable_pruner:
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val_acc


@section_printer("Hyperparameter Tuning")
def main() -> None:
    """Run automated Optuna hyperparameter optimization sweep."""
    args = parse_tune_args()

    # Determine compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing search on: {device}")
    print(
        f"Search Config: {args.n_trials} trials | "
        f"{args.epochs_per_trial} epochs/trial | Pruning: {args.prune}"
    )

    # 1. Load MNIST Dataset
    raw_dataset = load_mnist_data(RANDOM_SEED, normalize=True)
    dataset_tensors = {
        "X_train": torch.tensor(raw_dataset.X_train, dtype=torch.float32),
        "X_val": torch.tensor(raw_dataset.X_val, dtype=torch.float32),
        "X_test": torch.tensor(raw_dataset.X_test, dtype=torch.float32),
        "y_train": torch.tensor(raw_dataset.y_train, dtype=torch.long),
        "y_val": torch.tensor(raw_dataset.y_val, dtype=torch.long),
        "y_test": torch.tensor(raw_dataset.y_test, dtype=torch.long),
    }

    # 2. Setup Study Directory
    timestamp = str(int(time.time()))
    study_dir = BASE_ARTIFACT_DIRECTORY / f"tune_{timestamp}"
    study_dir.mkdir(parents=True, exist_ok=True)

    # 3. Create Optuna Study with TPE Sampler and Median Pruner
    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2) if args.prune else None

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # 4. Execute Optimization Sweep
    def _wrapped_objective(trial: optuna.Trial) -> float:
        return objective(
            trial=trial,
            dataset_tensors=dataset_tensors,
            image_shape=raw_dataset.image_shape,
            classes=raw_dataset.classes,
            model_type_choice=args.model_type,
            epochs=args.epochs_per_trial,
            device=device,
            enable_pruner=args.prune,
        )

    study.optimize(_wrapped_objective, n_trials=args.n_trials, show_progress_bar=True)

    # 5. Log and Save Search Results
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n================ Study Complete ================")
    print(f"Total Trials: {len(study.trials)}")
    print(f"Complete Trials: {len(complete_trials)}")
    print(f"Pruned Trials: {len(pruned_trials)}")
    print(f"Best Validation Accuracy: {study.best_value * 100:.2f}%")
    print(f"Best Hyperparameters: {json.dumps(study.best_params, indent=2)}")
    print("================================================")

    # Save Best Parameters
    best_summary = {
        "best_trial_number": study.best_trial.number,
        "best_val_accuracy": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "model_type": args.model_type,
    }
    with open(study_dir / "best_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    # Save Complete Trials Dataframe
    df_trials = study.trials_dataframe()
    df_trials.to_csv(study_dir / "tuning_trials.csv", index=False)

    # 6. Train Final Best Model with Full Epochs (e.g. 15 Epochs) & Save Artifacts
    print("\nTraining Final Candidate Model using Best Hyperparameters...")
    best_p = study.best_params

    final_model_type = best_p.get("model_type", args.model_type)
    if final_model_type == "cnn":
        channel_map = {
            "small_16_32": [16, 32],
            "medium_32_64": [32, 64],
            "large_32_64_128": [32, 64, 128],
        }
        conv_ch = channel_map[best_p.get("conv_config", "medium_32_64")]
        fc_h = best_p.get("fc_hidden", 128)
        final_model = CNNClassifier(
            in_dims=raw_dataset.image_shape,
            conv_channels=conv_ch,
            fc_hidden=fc_h,
            num_classes=len(raw_dataset.classes),
            dropout=best_p.get("dropout", 0.2),
        ).to(device)
    else:
        hidden_map = {
            "128_64": [128, 64],
            "256_128": [256, 128],
            "256_128_64": [256, 128, 64],
        }
        hid = hidden_map[best_p.get("hidden_config", "128_64")]
        final_model = MLPClassifier(
            in_dim=784,
            out_dim=len(raw_dataset.classes),
            hidden=hid,
            dropout=best_p.get("dropout", 0.0),
            use_batchnorm=best_p.get("use_batchnorm", False),
        ).to(device)

    # Save Final Model Checkpoint
    checkpoint_path = study_dir / "best_tuned_model.pt"
    torch.save(
        {
            "model_meta": {
                "class_name": final_model.__class__.__name__,
                "module_path": final_model.__class__.__module__,
                "init_args": best_p,
            },
            "model_state": final_model.state_dict(),
        },
        checkpoint_path,
    )

    print(f"\nAll tuning artifacts saved successfully to: {study_dir.resolve()}")


if __name__ == "__main__":
    main()
