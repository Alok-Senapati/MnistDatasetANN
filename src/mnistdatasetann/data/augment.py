"""Data augmentation transforms and dataset wrappers for MNIST training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


def get_train_transforms(
    degrees: float = 12.0,
    translate: float = 0.08,
    scale: tuple[float, float] = (0.92, 1.08),
) -> v2.Compose:
    """Build the stochastic data augmentation pipeline for training.

    Applies random affine transformations (rotation, translation, scaling) and
    random perspective distortion on the fly.

    Args:
        degrees: Maximum absolute rotation angle in degrees `(-degrees, +degrees)`.
            Defaults to `12.0`.
        translate: Maximum horizontal and vertical shift as fraction of image size.
            Defaults to `0.08`.
        scale: Minimum and maximum scaling factors `(min_scale, max_scale)`.
            Defaults to `(0.92, 1.08)`.

    Returns:
        A composable `torchvision.transforms.v2.Compose` pipeline.
    """
    return v2.Compose(
        [
            v2.RandomAffine(
                degrees=(-degrees, degrees), translate=(translate, translate), scale=scale
            ),
            v2.RandomPerspective(distortion_scale=0.15, p=0.25),
        ]
    )


class AugmentedDataset(Dataset):
    """Custom PyTorch dataset applying on-the-fly stochastic transformations.

    Takes flattened or 2D image tensors and applies the configured augmentation
    pipeline dynamically upon retrieval.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        transform: v2.Compose | None = None,
    ) -> None:
        """Initialize the augmented dataset wrapper.

        Args:
            features: 2D float tensor of flattened image samples `(N, 784)`.
            labels: 1D long tensor of target digit labels `(N,)`.
            transform: Optional torchvision transform pipeline to apply to each sample.
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and augment a single sample by its index.

        Args:
            index: Sample index.

        Returns:
            A tuple of `(augmented_features, label)` where features is a 1D tensor `(784,)`.
        """
        x = self.features[index].reshape(1, 28, 28)
        y = self.labels[index]

        if self.transform is not None:
            x = self.transform(x)

        return x.flatten(), y
