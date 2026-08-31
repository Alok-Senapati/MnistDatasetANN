"""Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN explainability."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) generator.

    Hooks into intermediate convolutional layers to compute coarse 2D localization
    heatmaps highlighting high-level discriminative regions used for prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        """Initialize Grad-CAM with forward and backward hooks on the target layer.

        Args:
            model: PyTorch neural network model (e.g. CNNClassifier).
            target_layer: The specific convolutional layer to hook into. If None,
                automatically defaults to the last convolutional block in `conv_layers`.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to the final block of the CNN's convolutional feature extractor
            if hasattr(model, "conv_layers") and len(model.conv_layers) > 0:
                target_layer = list(model.conv_layers.values())[-1]
            else:
                raise ValueError("No target convolutional layer provided or discovered in model.")

        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        # Register forward hook to capture feature activations A^k
        self.handles.append(self.target_layer.register_forward_hook(self._save_activation_hook))
        # Register full backward hook to capture gradients ∂S_c / ∂A^k
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradient_hook))

    def _save_activation_hook(
        self,
        module: nn.Module,
        input_tensor: tuple[torch.Tensor, ...],
        output_tensor: torch.Tensor,
    ) -> None:
        """Save intermediate feature map activations during forward pass."""
        self.activations = output_tensor.detach()

    def _save_gradient_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Save intermediate feature map gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        target_size: tuple[int, int] = (28, 28),
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for a sample input.

        Args:
            input_tensor: 1D `(784,)` or 4D `(1, 1, 28, 28)` input image tensor.
            target_class: Class index to explain. If None, uses model's predicted class.
            target_size: Output 2D dimensions `(H, W)` for bilinear upsampling.
                Defaults to `(28, 28)`.

        Returns:
            A 2D NumPy array of shape `(28, 28)` with normalized CAM values in `[0, 1]`.
        """
        if input_tensor.ndim == 1:
            x = input_tensor.unsqueeze(0).clone()
        else:
            x = input_tensor.clone()

        device = next(self.model.parameters()).device
        x = x.to(device)

        self.model.zero_grad()
        logits = self.model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # 1. Global Average Pooling of gradients -> importance weights alpha (B, C, 1, 1)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # 2. Weighted sum of activations across channels -> (B, 1, H', W')
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # 3. Apply ReLU to retain only positive contributions
        cam = F.relu(cam)

        # 4. Bilinear upsample to target image resolution (28x28)
        cam_upsampled = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)

        # 5. Normalize heatmap to [0, 1]
        cam_2d = cam_upsampled.squeeze().cpu().numpy()
        max_val = np.max(cam_2d)
        min_val = np.min(cam_2d)

        if max_val > min_val:
            cam_norm = (cam_2d - min_val) / (max_val - min_val + 1e-8)
        else:
            cam_norm = np.zeros_like(cam_2d)

        return cam_norm

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch forward and backward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __del__(self) -> None:
        """Ensure hooks are cleaned up upon garbage collection."""
        self.remove_hooks()
