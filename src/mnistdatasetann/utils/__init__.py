"""Utility helpers for printing, timing, plotting, and gradient diagnostics."""

from .diagnose import compute_gradient_norms
from .printer import section_printer
from .timer import Timer
from .visualizer import (
    visualize_accuracy,
    visualize_confusion_matrix,
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
    "compute_gradient_norms",
    "Timer",
]
