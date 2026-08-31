"""Training entry point for the MNIST MLP classifier."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from mnistdatasetann.args.args_class import TrainingArgs
from mnistdatasetann.data import get_train_transforms, load_mnist_data
from mnistdatasetann.model import (
    CNNClassifier,
    MLPClassifier,
    evaluate,
    get_optimizer,
    get_scheduler,
    train,
)
from mnistdatasetann.utils import (
    visualize_confusion_matrix,
    visualize_feature_maps,
    visualize_misclassified,
)

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = f"{RANDOM_SEED}"

BASE_ARTIFACT_DIRECTORY = Path(__file__).resolve().parents[1] / "artifacts"


def parse_cli_args() -> TrainingArgs:
    """Parse and validate the command-line options required for training.

    Returns:
        A populated :class:`TrainingArgs` instance matching the CLI contract for the
        training pipeline.
    """
    parser = argparse.ArgumentParser(
        description="MNIST classifier training entry point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type", type=str, choices=["mlp", "cnn"], default="mlp", help="Model Type"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adam",
        help="Optimizer type.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum used only for SGD optimization.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden-layer dimensions for the MLP.",
    )
    parser.add_argument(
        "--use-batchnorm",
        action="store_true",
        help="Enable batch normalization after each hidden layer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs without validation improvement before early stopping.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "step", "cosine", "plateau"],
        default="none",
        help="Learning rate scheduler family.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate floor for schedulers.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=5,
        help="Period of learning rate decay in epochs for StepLR.",
    )
    parser.add_argument(
        "--use-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable TensorBoard experiment tracking.",
    )
    parser.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=[32, 64],
        help="List of output channel depths for each convolutional block.",
    )
    parser.add_argument(
        "--fc-hidden",
        type=int,
        default=128,
        help="Number of hidden units in the dense classification head.",
    )
    parser.add_argument(
        "--use-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable stochastic on-the-fly data augmentation for training.",
    )
    parser.add_argument(
        "--augment-degrees",
        type=float,
        default=12.0,
        help="Maximum rotation angle in degrees for data augmentation.",
    )
    parser.add_argument(
        "--augment-translate",
        type=float,
        default=0.08,
        help="Maximum translation shift fraction for data augmentation.",
    )

    return parser.parse_args(namespace=TrainingArgs())


def main() -> None:
    """Run the full training pipeline for a single experiment.

    This function loads the dataset, prepares PyTorch data loaders, initializes the
    classifier, and launches the training loop while writing experiment artifacts.
    """
    args = parse_cli_args()

    # Keep the result deterministic across repeated runs for debugging and comparison.
    torch.random.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Each run gets a dedicated artifact directory so multiple experiments can coexist.
    run_id = str(int(time.time()))
    artifacts_dir = BASE_ARTIFACT_DIRECTORY / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_mnist_data(RANDOM_SEED, normalize=True)

    X_train = torch.tensor(dataset.X_train, dtype=torch.float32)
    X_val = torch.tensor(dataset.X_val, dtype=torch.float32)
    X_test = torch.tensor(dataset.X_test, dtype=torch.float32)
    y_train = torch.tensor(dataset.y_train, dtype=torch.long)
    y_val = torch.tensor(dataset.y_val, dtype=torch.long)
    y_test = torch.tensor(dataset.y_test, dtype=torch.long)

    n_features = dataset.num_flattened_features
    n_classes = len(dataset.classes)

    train_dataset = TensorDataset(X_train, y_train)

    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=256)
    _test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256)

    if args.model_type == "mlp":
        model_init_args = {
            "hidden": args.hidden,
            "in_dim": n_features,
            "out_dim": n_classes,
            "dropout": args.dropout,
            "use_batchnorm": args.use_batchnorm,
        }
    else:
        model_init_args = {
            "in_dims": dataset.image_shape,
            "conv_channels": args.conv_channels,
            "fc_hidden": args.fc_hidden,
            "num_classes": n_classes,
            "dropout": args.dropout,
        }

    model_class = MLPClassifier if args.model_type == "mlp" else CNNClassifier
    model = model_class(**model_init_args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay, args.momentum)
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        min_lr=args.min_lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_step_size=args.lr_step_size,
    )
    writer: SummaryWriter | None = None
    if args.use_tensorboard:
        writer = SummaryWriter(log_dir=str(artifacts_dir / "tensorboard"))

    gpu_transforms: nn.Module | None = None
    if args.use_augmentation:
        gpu_transforms = get_train_transforms(
            degrees=args.augment_degrees,
            translate=args.augment_translate,
            scale=(args.augment_scale_min, args.augment_scale_max),
        ).to(device)

    _, best_model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        model_init_args=model_init_args,
        device=device,
        lr_scheduler=lr_scheduler,
        output_path=artifacts_dir,
        patience=args.patience,
        use_tensorboard=args.use_tensorboard,
        writer=writer,
        transform=gpu_transforms,
    )

    results = evaluate(
        model=best_model, data_loader=_test_loader, device=device, classes=dataset.classes
    )

    visualize_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        classes=dataset.classes,
        save_path=artifacts_dir / "confusion_matrix.png",
    )

    visualize_misclassified(
        images=results["images"],
        y_true=results["y_true"],
        y_pred=results["y_pred"],
        y_probs=results["y_probs"],
        max_samples=20,
        save_path=artifacts_dir / "misclassified_samples.png",
    )

    if hasattr(best_model, "get_feature_maps"):
        sample_tensor = X_test[:1].to(device)
        fmaps = best_model.get_feature_maps(sample_tensor)
        fig_fmaps = visualize_feature_maps(
            feature_maps=fmaps,
            raw_image=X_test[0].numpy(),
            max_channels_per_layer=64,
            save_path=artifacts_dir / "conv_feature_maps.png",
        )
        if writer is not None:
            writer.add_figure("diagnostics/conv_feature_maps", fig_fmaps)

    with open(artifacts_dir / "training_args.json", "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)

    with open(artifacts_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(results["report_dict"], f, indent=2)

    if writer is not None:
        writer.add_hparams(
            hparam_dict={
                "model_type": args.model_type,
                "use_augmentation": args.use_augmentation,
                "lr": args.lr,
                "optimizer": args.optimizer,
                "batch_size": args.batch_size,
                "lr_scheduler": args.scheduler,
                "use_batchnorm": args.use_batchnorm,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "lr_decay_factor": args.lr_decay_factor,
                "lr_step_size": args.lr_step_size,
            },
            metric_dict={
                "eval/test_accuracy": float(results["report_dict"]["accuracy"]),
            },
        )
        writer.close()


if __name__ == "__main__":
    main()
