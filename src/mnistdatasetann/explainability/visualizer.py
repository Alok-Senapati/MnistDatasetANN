"""Visualization utilities for Explainable AI (Saliency Maps and Grad-CAM)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_explanations(
    raw_image: np.ndarray | torch.Tensor,
    saliency_map: np.ndarray | None = None,
    gradcam_map: np.ndarray | None = None,
    predicted_class: int | None = None,
    confidence: float | None = None,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot original digit alongside Saliency, Grad-CAM, and overlay attention heatmaps.

    Args:
        raw_image: 1D `(784,)` or 2D `(28, 28)` raw image array.
        saliency_map: Optional 2D normalized saliency map `(28, 28)`.
        gradcam_map: Optional 2D normalized Grad-CAM heatmap `(28, 28)`.
        predicted_class: Optional predicted discrete class index.
        confidence: Optional prediction confidence percentage.
        save_path: Optional file path to save the generated PNG plot.

    Returns:
        The matplotlib `Figure` containing the multi-panel explanation grid.
    """
    if isinstance(raw_image, torch.Tensor):
        base_arr = raw_image.detach().cpu().numpy()
    else:
        base_arr = np.asarray(raw_image)

    if base_arr.ndim == 1:
        base_arr = base_arr.reshape((28, 28))
    elif base_arr.ndim == 3 and base_arr.shape[0] == 1:
        base_arr = base_arr[0]

    # Build active panel list
    panels: list[tuple[str, np.ndarray, str, bool]] = [("Input Image", base_arr, "gray", False)]

    if saliency_map is not None:
        panels.append(("Vanilla Saliency\n(Pixel Gradients)", saliency_map, "hot", False))

    if gradcam_map is not None:
        panels.append(("Grad-CAM Heatmap\n(Filter Attention)", gradcam_map, "jet", False))
        # Add blended overlay
        panels.append(("Attention Overlay", gradcam_map, "jet", True))

    num_panels = len(panels)
    fig, axes = plt.subplots(1, num_panels, figsize=(3.2 * num_panels, 3.6))

    if num_panels == 1:
        axes = [axes]

    title_suffix = ""
    if predicted_class is not None:
        title_suffix = f" — Pred: {predicted_class}"
        if confidence is not None:
            title_suffix += f" ({confidence:.1f}%)"

    for idx, (title, img_data, cmap, is_overlay) in enumerate(panels):
        ax = axes[idx]
        if is_overlay:
            # Base grayscale image
            ax.imshow(base_arr, cmap="gray")
            # Alpha-blended Grad-CAM heatmap
            im = ax.imshow(img_data, cmap=cmap, alpha=0.55)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            im = ax.imshow(img_data, cmap=cmap)
            if cmap != "gray":
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Model Explainability Dashboard{title_suffix}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        target = Path(save_path)
        if target.suffix == "":
            target = target.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig
