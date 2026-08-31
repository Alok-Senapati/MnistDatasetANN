"""Pixel-level vanilla saliency map generation via input gradients."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_saliency_map(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Compute normalized pixel-level saliency map using first-order input gradients.

    Args:
        model: PyTorch neural network model in evaluation mode.
        input_tensor: 1D `(784,)` or 4D `(1, 1, 28, 28)` input image tensor.
        target_class: Target class index to explain. If None, uses the model's predicted class.

    Returns:
        A 2D NumPy array `(28, 28)` with normalized saliency values in `[0, 1]`.
    """
    model.eval()

    if input_tensor.ndim == 1:
        x = input_tensor.unsqueeze(0).clone().detach()
    else:
        x = input_tensor.clone().detach()

    x.requires_grad_(True)

    logits: torch.Tensor = model(x)

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    if x.grad is None:
        return np.zeros((28, 28), dtype=np.float32)

    gradients = x.grad.data.abs()

    saliency_2d = gradients.squeeze().cpu().numpy()
    if saliency_2d.ndim == 3:
        saliency_2d = saliency_2d[0]
    elif saliency_2d.ndim == 1:
        saliency_2d = saliency_2d.reshape(28, 28)

    max_val = np.max(saliency_2d)
    min_val = np.min(saliency_2d)
    if max_val > min_val:
        saliency_norm = (saliency_2d - min_val) / (max_val - min_val + 1e-8)
    else:
        saliency_norm = np.zeros_like(saliency_2d)

    return saliency_norm
