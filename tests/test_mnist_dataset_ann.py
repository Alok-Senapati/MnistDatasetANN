"""Unit tests for the MNIST ANN project."""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from mnistdatasetann.data import load_mnist_data
from mnistdatasetann.model import MLPClassifier, get_optimizer, get_scheduler
from mnistdatasetann.utils.printer import section_printer
from mnistdatasetann.utils.timer import Timer
from mnistdatasetann.utils.visualizer import visualize_accuracy, visualize_loss, visualize_lr


class TestSectionPrinter(unittest.TestCase):
    """Validate the console decoration helper used around data and training sections."""

    def test_section_printer_wraps_function(self) -> None:
        @section_printer("Demo Section")
        def sample(value: str) -> str:
            return value.upper()

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = sample("hello")

        self.assertEqual(result, "HELLO")
        output = stdout.getvalue()
        self.assertIn("Demo Section", output)
        self.assertIn("=", output)


class TestTimer(unittest.TestCase):
    """Validate the elapsed-time context manager."""

    def test_timer_records_elapsed_time(self) -> None:
        with Timer() as timer:
            time.sleep(0.01)

        self.assertGreaterEqual(timer.elapsed, 0.0)
        self.assertIsNotNone(timer.start_time)


class TestMLPClassifier(unittest.TestCase):
    """High-level smoke tests for the classifier architecture and outputs."""

    def test_model_forward_and_predictions(self) -> None:
        model = MLPClassifier(
            hidden=[16, 8],
            in_dim=4,
            out_dim=3,
            dropout=0.0,
            use_batchnorm=False,
        )

        inputs = torch.randn(5, 4)
        probabilities = model.predict_proba(inputs)
        predictions = model.predict(inputs)

        self.assertEqual(probabilities.shape, (5, 3))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-5))
        self.assertEqual(predictions.shape, (5,))


class TestOptimizerFactory(unittest.TestCase):
    """Confirm the optimizer factory returns the expected backend objects."""

    def test_supported_optimizers(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)

        adam = get_optimizer(model, "adam", 1e-3, 0.0, 0.9)
        adamw = get_optimizer(model, "adamw", 1e-3, 0.0, 0.9)
        sgd = get_optimizer(model, "sgd", 1e-3, 0.0, 0.9)

        self.assertIsInstance(adam, torch.optim.Adam)
        self.assertIsInstance(adamw, torch.optim.AdamW)
        self.assertIsInstance(sgd, torch.optim.SGD)

    def test_invalid_optimizer_raises_value_error(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)

        with self.assertRaises(ValueError):
            get_optimizer(model, "unsupported", 1e-3, 0.0, 0.9)


class TestSchedulerFactory(unittest.TestCase):
    """Confirm the scheduler factory returns the expected scheduler objects."""

    def test_supported_schedulers(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)
        optimizer = get_optimizer(model, "adam", 1e-3, 0.0, 0.9)

        step = get_scheduler(optimizer, "step", 1e-6, 0.1, 5)
        cosine = get_scheduler(optimizer, "cosine", 1e-6, 0.1, 5)
        plateau = get_scheduler(optimizer, "plateau", 1e-6, 0.1, 5)
        none = get_scheduler(optimizer, "none", 1e-6, 0.1, 5)

        self.assertIsInstance(step, torch.optim.lr_scheduler.StepLR)
        self.assertIsInstance(cosine, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertIsInstance(plateau, torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.assertIsNone(none)


class TestMnistDataLoader(unittest.TestCase):
    """Smoke-test the real CSV data loading path used by the training pipeline."""

    def test_load_mnist_data_smoke(self) -> None:
        dataset = load_mnist_data(random_seed=7, normalize=True)

        self.assertEqual(dataset.X_train.shape[1], 784)
        self.assertEqual(dataset.X_val.shape[1], 784)
        self.assertEqual(dataset.X_test.shape[1], 784)
        self.assertEqual(dataset.y_train.shape[0], dataset.X_train.shape[0])
        self.assertTrue(dataset.is_normalized)
        self.assertGreater(len(dataset.classes), 0)


class TestVisualizationHelpers(unittest.TestCase):
    """Ensure metric plots can be rendered and saved without runtime errors."""

    def test_visualizers_can_write_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            loss_path = output_dir / "loss_plot"
            accuracy_path = output_dir / "accuracy_plot"
            lr_path = output_dir / "lr_plot"

            visualize_loss([1.0, 0.7, 0.5], [1.2, 0.9, 0.6], save_path=loss_path)
            visualize_accuracy([0.5, 0.7, 0.8], [0.4, 0.65, 0.75], save_path=accuracy_path)
            visualize_lr([1e-3, 5e-4, 1e-4], save_path=lr_path)

            self.assertTrue((loss_path.with_suffix(".png")).exists())
            self.assertTrue((accuracy_path.with_suffix(".png")).exists())
            self.assertTrue((lr_path.with_suffix(".png")).exists())


if __name__ == "__main__":
    unittest.main()
