"""Tests for src.neural_net module."""

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from src.neural_net import BiologicalClassifier, ConvolutionalClassifier, ModelTrainer


@pytest.fixture
def synthetic_data():
    """Small reproducible dataset for neural net tests."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 32)).astype(np.float32)
    y = rng.integers(0, 2, size=100)
    return X, y


class TestBiologicalClassifier:
    def test_forward_shape(self):
        model = BiologicalClassifier(input_dim=32, hidden_dims=[16, 8], output_dim=2)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 2)

    def test_custom_hidden_dims(self):
        model = BiologicalClassifier(input_dim=16, hidden_dims=[64], output_dim=3)
        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 3)

    def test_parameters_learnable(self):
        model = BiologicalClassifier(input_dim=10, hidden_dims=[8], output_dim=2)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


class TestConvolutionalClassifier:
    def test_forward_shape(self):
        # input_dim must be divisible by 4 after two MaxPool1d(2) layers
        model = ConvolutionalClassifier(input_dim=32, output_dim=2, n_filters=8)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 2)

    def test_multiclass_output(self):
        model = ConvolutionalClassifier(input_dim=64, output_dim=5, n_filters=16)
        x = torch.randn(3, 64)
        out = model(x)
        assert out.shape == (3, 5)


class TestModelTrainer:
    def test_train_runs(self, synthetic_data):
        X, y = synthetic_data
        model = BiologicalClassifier(input_dim=X.shape[1], hidden_dims=[8], output_dim=2)
        trainer = ModelTrainer(model, device="cpu", learning_rate=0.01)
        trainer.train(X, y, epochs=2, batch_size=16, validation_split=0.2)
        assert len(trainer.history["train_loss"]) == 2
        assert len(trainer.history["val_loss"]) == 2
        assert len(trainer.history["val_acc"]) == 2

    def test_predict_returns_labels(self, synthetic_data):
        X, y = synthetic_data
        model = BiologicalClassifier(input_dim=X.shape[1], hidden_dims=[8], output_dim=2)
        trainer = ModelTrainer(model, device="cpu")
        trainer.train(X, y, epochs=1, batch_size=16)
        preds = trainer.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, synthetic_data):
        X, y = synthetic_data
        model = BiologicalClassifier(input_dim=X.shape[1], hidden_dims=[8], output_dim=2)
        trainer = ModelTrainer(model, device="cpu")
        trainer.train(X, y, epochs=1, batch_size=16)
        proba = trainer.predict(X, return_proba=True)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_save_and_load_roundtrip(self, synthetic_data, tmp_path):
        X, y = synthetic_data
        model = BiologicalClassifier(input_dim=X.shape[1], hidden_dims=[8], output_dim=2)
        trainer = ModelTrainer(model, device="cpu")
        trainer.train(X, y, epochs=1, batch_size=16)

        ckpt = tmp_path / "ckpt.pt"
        trainer.save(str(ckpt))
        assert ckpt.exists()

        model2 = BiologicalClassifier(input_dim=X.shape[1], hidden_dims=[8], output_dim=2)
        trainer2 = ModelTrainer(model2, device="cpu")
        trainer2.load(str(ckpt))
        # Predictions should match after reloading
        np.testing.assert_array_equal(trainer.predict(X), trainer2.predict(X))
