"""Utility helpers for printing, timing, plotting, and gradient diagnostics."""

from .diagnose import compute_gradient_norms
from .preprocessor import preprocess_image
from .printer import section_printer
from .timer import Timer
from .visualizer import (
    visualize_accuracy,
    visualize_confusion_matrix,
    visualize_feature_maps,
    visualize_loss,
    visualize_lr,
    visualize_misclassified,
)

__all__ = [
    "section_printer",
    "visualize_loss",
    "visualize_accuracy",
    "visualize_lr",
    "visualize_confusion_matrix",
    "visualize_misclassified",
    "visualize_feature_maps",
    "compute_gradient_norms",
    "preprocess_image",
    "Timer",
]
