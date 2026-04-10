"""Tests for src.evaluation module."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.evaluation import ModelEvaluator
from src.models import compare_models, evaluate_model


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 10))
    # Easily separable target
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def fitted_model(binary_data):
    X, y = binary_data
    model = LogisticRegression(max_iter=500).fit(X, y)
    return model


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class TestModelEvaluator:
    def test_plot_roc_curve_returns_auc(self, binary_data, fitted_model, tmp_path):
        X, y = binary_data
        y_proba = fitted_model.predict_proba(X)
        evaluator = ModelEvaluator()
        auc_value = evaluator.plot_roc_curve(y, y_proba, save_path=str(tmp_path / "roc.png"))
        assert 0.0 <= auc_value <= 1.0
        assert (tmp_path / "roc.png").exists()

    def test_plot_confusion_matrix_returns_cm(self, binary_data, fitted_model, tmp_path):
        X, y = binary_data
        y_pred = fitted_model.predict(X)
        evaluator = ModelEvaluator()
        cm = evaluator.plot_confusion_matrix(y, y_pred, save_path=str(tmp_path / "cm.png"))
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y)

    def test_plot_precision_recall_curve(self, binary_data, fitted_model, tmp_path):
        X, y = binary_data
        y_proba = fitted_model.predict_proba(X)
        evaluator = ModelEvaluator()
        evaluator.plot_precision_recall_curve(y, y_proba, save_path=str(tmp_path / "pr.png"))
        assert (tmp_path / "pr.png").exists()

    def test_feature_importance_raises_without_attr(self, fitted_model):
        # LogisticRegression has no feature_importances_
        evaluator = ModelEvaluator()
        with pytest.raises(ValueError):
            evaluator.plot_feature_importance(fitted_model, [f"f{i}" for i in range(10)])

    def test_generate_report_returns_string(self, binary_data, fitted_model):
        X, y = binary_data
        y_pred = fitted_model.predict(X)
        evaluator = ModelEvaluator()
        report = evaluator.generate_report(y, y_pred, class_names=["neg", "pos"])
        assert isinstance(report, str)
        assert "neg" in report and "pos" in report


# ---------------------------------------------------------------------------
# compare_models (imported from src.models, previously duplicated)
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_compare_models_returns_dataframe(self):
        results = {
            "model_a": {"accuracy_mean": 0.8, "f1_mean": 0.79, "roc_auc_mean": 0.85,
                        "accuracy_std": 0.02, "f1_std": 0.02, "roc_auc_std": 0.01},
            "model_b": {"accuracy_mean": 0.9, "f1_mean": 0.91, "roc_auc_mean": 0.93,
                        "accuracy_std": 0.01, "f1_std": 0.01, "roc_auc_std": 0.01},
        }
        summary = compare_models(results)
        assert len(summary) == 2
        # Should be sorted by f1_mean descending
        assert summary.index[0] == "model_b"

    def test_compare_models_with_evaluate_model_output(self, binary_data, fitted_model):
        X, y = binary_data
        metrics = evaluate_model(fitted_model, X, y)
        summary = compare_models({"logreg": metrics})
        assert "logreg" in summary.index
        for key in ("accuracy", "precision", "recall", "f1", "auc"):
            assert key in summary.columns
