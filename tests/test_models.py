"""
Tests for classical ML model training and evaluation.
"""

import numpy as np
import pytest

from src.models import (
    compare_models,
    evaluate_model,
    train_elastic_net,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
)


def _make_dataset(n_samples=200, n_features=20, n_classes=2, seed=42):
    """Generate a simple synthetic classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)
    return X, y


class TestTrainFunctions:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.X, self.y = _make_dataset()

    def test_logistic_regression_returns_metrics(self):
        model, metrics = train_logistic_regression(self.X, self.y, cv_folds=3)
        assert "accuracy_mean" in metrics
        assert "f1_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0

    def test_random_forest_returns_metrics(self):
        model, metrics = train_random_forest(self.X, self.y, cv_folds=3)
        assert "accuracy_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0

    def test_gradient_boosting_returns_metrics(self):
        model, metrics = train_gradient_boosting(self.X, self.y, cv_folds=3)
        assert "accuracy_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0

    def test_elastic_net_returns_metrics(self):
        model, metrics = train_elastic_net(self.X, self.y, cv_folds=3)
        assert "accuracy_mean" in metrics
        assert 0.0 <= metrics["accuracy_mean"] <= 1.0


class TestEvaluateModel:
    def test_evaluate_returns_all_metrics(self):
        X, y = _make_dataset()
        model, _ = train_logistic_regression(X, y, cv_folds=3)
        metrics = evaluate_model(model, X, y)
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in metrics
            assert isinstance(metrics[key], float)


class TestCompareModels:
    def test_compare_returns_dataframe(self):
        results = {
            "lr": {"f1": 0.80, "accuracy": 0.82},
            "rf": {"f1": 0.85, "accuracy": 0.86},
        }
        df = compare_models(results)
        assert df.shape[0] == 2
        assert df.index[0] == "rf"  # sorted by f1 descending
