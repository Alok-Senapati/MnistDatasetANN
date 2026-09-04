"""Unit tests for the MNIST ANN project."""

from __future__ import annotations

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mnistdatasetann.data import AugmentedDataset, get_train_transforms, load_mnist_data
from mnistdatasetann.explainability import (
    GradCAM,
    compute_saliency_map,
    visualize_explanations,
)
from mnistdatasetann.model import (
    CNNClassifier,
    MLPClassifier,
    evaluate,
    get_optimizer,
    get_scheduler,
    load_model,
)
from mnistdatasetann.utils.diagnose import compute_gradient_norms
from mnistdatasetann.utils.preprocessor import preprocess_image
from mnistdatasetann.utils.printer import section_printer
from mnistdatasetann.utils.timer import Timer
from mnistdatasetann.utils.visualizer import (
    visualize_accuracy,
    visualize_augmentations,
    visualize_confusion_matrix,
    visualize_feature_maps,
    visualize_loss,
    visualize_lr,
    visualize_misclassified,
)


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


class TestDiagnostics(unittest.TestCase):
    """Validate gradient norms computation helper."""

    def test_compute_gradient_norms_with_gradients(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)
        inputs = torch.randn(2, 4)
        targets = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()

        norms = compute_gradient_norms(model)
        self.assertIn("grad_norm/total", norms)
        self.assertGreater(norms["grad_norm/total"], 0.0)
        self.assertTrue(any("head.weight" in k for k in norms))


class TestPreprocessor(unittest.TestCase):
    """Validate image preprocessing pipeline for arbitrary-resolution inputs."""

    def test_preprocess_image_otsu_and_standard_modes(self) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for x in range(30, 70):
            for y in range(30, 70):
                img.putpixel((x, y), (0, 0, 0))

        # Test Otsu mode
        features_otsu, canvas_otsu = preprocess_image(img, auto_invert=True, method="otsu")
        self.assertEqual(features_otsu.shape, (1, 784))
        self.assertEqual(canvas_otsu.size, (28, 28))
        self.assertGreaterEqual(float(features_otsu.min()), 0.0)
        self.assertLessEqual(float(features_otsu.max()), 1.0)

        # Test Standard mode
        features_std, canvas_std = preprocess_image(img, auto_invert=True, method="standard")
        self.assertEqual(features_std.shape, (1, 784))
        self.assertEqual(canvas_std.size, (28, 28))
        self.assertGreaterEqual(float(features_std.min()), 0.0)
        self.assertLessEqual(float(features_std.max()), 1.0)


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


class TestCNNClassifier(unittest.TestCase):
    """Smoke and behavioral tests for the CNNClassifier architecture and outputs."""

    def test_cnn_forward_with_4d_and_2d_inputs(self) -> None:
        model = CNNClassifier(
            in_dims=(1, 28, 28),
            conv_channels=(8, 16),
            fc_hidden=32,
            num_classes=10,
            dropout=0.1,
        )

        inputs_4d = torch.randn(4, 1, 28, 28)
        inputs_2d = torch.randn(4, 784)

        logits_4d = model(inputs_4d)
        logits_2d = model(inputs_2d)
        probabilities = model.predict_proba(inputs_2d)
        predictions = model.predict(inputs_2d)

        self.assertEqual(logits_4d.shape, (4, 10))
        self.assertEqual(logits_2d.shape, (4, 10))
        self.assertEqual(probabilities.shape, (4, 10))
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(4), atol=1e-5))
        self.assertEqual(predictions.shape, (4,))

    def test_cnn_get_feature_maps(self) -> None:
        model = CNNClassifier(
            in_dims=(1, 28, 28),
            conv_channels=(8, 16),
            fc_hidden=32,
            num_classes=10,
        )
        inputs = torch.randn(2, 1, 28, 28)
        fmaps = model.get_feature_maps(inputs)

        self.assertIn("conv1", fmaps)
        self.assertIn("conv2", fmaps)
        self.assertEqual(fmaps["conv1"].shape, (2, 8, 14, 14))
        self.assertEqual(fmaps["conv2"].shape, (2, 16, 7, 7))


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


class TestModelLoader(unittest.TestCase):
    """Validate loading saved model checkpoints from disk."""

    def test_save_and_load_model_matches_outputs(self) -> None:
        init_args = {
            "hidden": [16, 8],
            "in_dim": 4,
            "out_dim": 3,
            "dropout": 0.0,
            "use_batchnorm": False,
        }
        original_model = MLPClassifier(**init_args)
        inputs = torch.randn(3, 4)
        expected_preds = original_model.predict(inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "model.pt"
            checkpoint = {
                "model_meta": {
                    "class_name": original_model.__class__.__name__,
                    "module_path": original_model.__class__.__module__,
                    "init_args": init_args,
                },
                "model_state": original_model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            loaded_model = load_model(checkpoint_path)
            loaded_preds = loaded_model.predict(inputs)

            self.assertTrue(torch.equal(expected_preds, loaded_preds))

    def test_save_and_load_cnn_model_matches_outputs(self) -> None:
        init_args = {
            "in_dims": (1, 28, 28),
            "conv_channels": [8, 16],
            "fc_hidden": 32,
            "num_classes": 10,
            "dropout": 0.0,
        }
        original_model = CNNClassifier(**init_args).eval()
        inputs = torch.randn(3, 784)
        expected_preds = original_model.predict(inputs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "cnn_model.pt"
            checkpoint = {
                "model_meta": {
                    "class_name": original_model.__class__.__name__,
                    "module_path": original_model.__class__.__module__,
                    "init_args": init_args,
                },
                "model_state": original_model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            loaded_model = load_model(checkpoint_path)
            loaded_preds = loaded_model.predict(inputs)

            self.assertTrue(torch.equal(expected_preds, loaded_preds))

    def test_load_model_nonexistent_file_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model("nonexistent_checkpoint.pt")


class TestEvaluate(unittest.TestCase):
    """Validate model evaluation and metrics generation."""

    def test_evaluate_returns_expected_structure(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)
        features = torch.randn(10, 4)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        dataset = torch.utils.data.TensorDataset(features, targets)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        result = evaluate(model, data_loader, device="cpu", classes=[0, 1, 2])

        self.assertIn("images", result)
        self.assertIn("y_true", result)
        self.assertIn("y_pred", result)
        self.assertIn("y_probs", result)
        self.assertIn("report_dict", result)
        self.assertIn("report_str", result)

        self.assertEqual(result["images"].shape, (10, 4))
        self.assertEqual(result["y_true"].shape, (10,))
        self.assertEqual(result["y_pred"].shape, (10,))
        self.assertEqual(result["y_probs"].shape, (10, 3))
        self.assertIn("accuracy", result["report_dict"])

    def test_evaluate_empty_loader_raises_error(self) -> None:
        model = MLPClassifier(hidden=[8], in_dim=4, out_dim=3, dropout=0.0, use_batchnorm=False)
        empty_dataset = torch.utils.data.TensorDataset(
            torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        )
        empty_loader = torch.utils.data.DataLoader(empty_dataset, batch_size=4)

        with self.assertRaises(ValueError):
            evaluate(model, empty_loader)


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
            cm_path = output_dir / "cm_plot"
            misclassified_path = output_dir / "misclassified_plot"

            visualize_loss([1.0, 0.7, 0.5], [1.2, 0.9, 0.6], save_path=loss_path)
            visualize_accuracy([0.5, 0.7, 0.8], [0.4, 0.65, 0.75], save_path=accuracy_path)
            visualize_lr([1e-3, 5e-4, 1e-4], save_path=lr_path)
            visualize_confusion_matrix(
                [0, 1, 2, 1], [0, 2, 2, 1], classes=[0, 1, 2], save_path=cm_path
            )
            visualize_misclassified(
                images=np.random.rand(4, 784),
                y_true=np.array([0, 1, 2, 3]),
                y_pred=np.array([0, 2, 2, 1]),
                y_probs=np.array(
                    [
                        [0.9, 0.1, 0.0, 0.0],
                        [0.1, 0.2, 0.7, 0.0],
                        [0.1, 0.1, 0.8, 0.0],
                        [0.1, 0.6, 0.1, 0.2],
                    ]
                ),
                save_path=misclassified_path,
            )

            fmaps_path = output_dir / "fmaps_plot"
            fmaps_dummy = {
                "conv1": torch.randn(1, 4, 14, 14),
                "conv2": torch.randn(1, 8, 7, 7),
            }
            visualize_feature_maps(
                feature_maps=fmaps_dummy,
                raw_image=np.random.rand(28, 28),
                save_path=fmaps_path,
            )

            aug_path = output_dir / "aug_plot"
            visualize_augmentations(
                sample_image=np.random.rand(28, 28),
                transform=get_train_transforms(degrees=10.0),
                num_variations=4,
                save_path=aug_path,
            )

            self.assertTrue((loss_path.with_suffix(".png")).exists())
            self.assertTrue((accuracy_path.with_suffix(".png")).exists())
            self.assertTrue((lr_path.with_suffix(".png")).exists())
            self.assertTrue((cm_path.with_suffix(".png")).exists())
            self.assertTrue((misclassified_path.with_suffix(".png")).exists())
            self.assertTrue((fmaps_path.with_suffix(".png")).exists())
            self.assertTrue((aug_path.with_suffix(".png")).exists())


class TestAugmentation(unittest.TestCase):
    """Validate data augmentation pipeline and AugmentedDataset wrapper."""

    def test_get_train_transforms_creates_pipeline(self) -> None:
        transforms = get_train_transforms(degrees=15.0, translate=0.1, scale=(0.9, 1.1))
        dummy_sample = torch.randn(1, 28, 28)
        augmented = transforms(dummy_sample)

        self.assertEqual(augmented.shape, (1, 28, 28))

    def test_augmented_dataset_retrieval(self) -> None:
        features = torch.randn(10, 784)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        transforms = get_train_transforms(degrees=10.0)

        dataset = AugmentedDataset(features, labels, transform=transforms)

        self.assertEqual(len(dataset), 10)
        x_aug, y_val = dataset[0]
        self.assertEqual(x_aug.shape, (784,))
        self.assertEqual(int(y_val), 0)


class TestExplainability(unittest.TestCase):
    """Validate Saliency Maps and Grad-CAM explainability generators."""

    def test_saliency_map_generation(self) -> None:
        model = CNNClassifier(
            in_dims=(1, 28, 28),
            conv_channels=(8, 16),
            fc_hidden=32,
            num_classes=10,
        )
        dummy_input = torch.randn(1, 1, 28, 28)
        saliency = compute_saliency_map(model, dummy_input)

        self.assertEqual(saliency.shape, (28, 28))
        self.assertGreaterEqual(float(saliency.min()), 0.0)
        self.assertLessEqual(float(saliency.max()), 1.0 + 1e-6)

    def test_gradcam_generation_and_cleanup(self) -> None:
        model = CNNClassifier(
            in_dims=(1, 28, 28),
            conv_channels=(8, 16),
            fc_hidden=32,
            num_classes=10,
        )
        dummy_input = torch.randn(1, 1, 28, 28)
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(dummy_input, target_class=3)

        self.assertEqual(cam.shape, (28, 28))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)

        gradcam.remove_hooks()
        self.assertEqual(len(gradcam.handles), 0)

    def test_visualize_explanations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "explain_plot"
            fig = visualize_explanations(
                raw_image=np.random.rand(28, 28),
                saliency_map=np.random.rand(28, 28),
                gradcam_map=np.random.rand(28, 28),
                predicted_class=7,
                confidence=98.5,
                save_path=out_path,
            )
            self.assertIsNotNone(fig)
            self.assertTrue(out_path.with_suffix(".png").exists())


class TestTuneV2(unittest.TestCase):
    """Validate hyperparameter tuning pipeline utilities in tuneV2."""

    def test_parse_tune_args(self) -> None:
        from scripts.tuneV2 import parse_tune_args

        default_args = parse_tune_args([])
        self.assertEqual(default_args.model_type, "cnn")
        self.assertEqual(default_args.n_trials, 15)
        self.assertEqual(default_args.epochs_per_trial, 8)
        self.assertTrue(default_args.prune)

        custom_args = parse_tune_args(
            [
                "--model-type",
                "mlp",
                "--n-trials",
                "3",
                "--epochs-per-trial",
                "2",
                "--study-name",
                "test_study",
                "--storage",
                "sqlite:///test.db",
                "--no-prune",
            ]
        )
        self.assertEqual(custom_args.model_type, "mlp")
        self.assertEqual(custom_args.n_trials, 3)
        self.assertEqual(custom_args.epochs_per_trial, 2)
        self.assertEqual(custom_args.study_name, "test_study")
        self.assertEqual(custom_args.storage, "sqlite:///test.db")
        self.assertFalse(custom_args.prune)

    def test_objective_smoke_run_mlp(self) -> None:
        import optuna

        from scripts.tuneV2 import objective

        dataset_tensors = {
            "X_train": torch.randn(16, 784),
            "X_val": torch.randn(8, 784),
            "y_train": torch.randint(0, 10, (16,), dtype=torch.long),
            "y_val": torch.randint(0, 10, (8,), dtype=torch.long),
        }

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial=trial,
                dataset_tensors=dataset_tensors,
                image_shape=(1, 28, 28),
                classes=list(range(10)),
                model_type_choice="mlp",
                epochs=1,
                device="cpu",
                enable_pruner=False,
            ),
            n_trials=1,
        )
        self.assertEqual(len(study.trials), 1)
        self.assertGreaterEqual(study.best_value, 0.0)

    def test_objective_smoke_run_cnn(self) -> None:
        import optuna

        from scripts.tuneV2 import objective

        dataset_tensors = {
            "X_train": torch.randn(8, 784),
            "X_val": torch.randn(4, 784),
            "y_train": torch.randint(0, 10, (8,), dtype=torch.long),
            "y_val": torch.randint(0, 10, (4,), dtype=torch.long),
        }

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(
                trial=trial,
                dataset_tensors=dataset_tensors,
                image_shape=(1, 28, 28),
                classes=list(range(10)),
                model_type_choice="cnn",
                epochs=1,
                device="cpu",
                enable_pruner=False,
            ),
            n_trials=1,
        )
        self.assertEqual(len(study.trials), 1)
        self.assertGreaterEqual(study.best_value, 0.0)


if __name__ == "__main__":
    unittest.main()
