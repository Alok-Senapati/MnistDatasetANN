"""Explainable AI (XAI) tools including Saliency Maps and Grad-CAM for MNIST."""

from __future__ import annotations

from .gradcam import GradCAM
from .saliency import compute_saliency_map
from .visualizer import visualize_explanations

__all__ = [
    "compute_saliency_map",
    "GradCAM",
    "visualize_explanations",
]
