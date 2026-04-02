"""
Tests for feature engineering utilities.
"""

import numpy as np
import pytest

from src.features import (
    FeatureEngineer,
    create_interaction_features,
    create_polynomial_features,
)


class TestFeatureEngineer:
    def test_scale_features_standard(self):
        fe = FeatureEngineer()
        X = np.random.RandomState(42).randn(50, 10)
        X_scaled = fe.scale_features(X, method="standard", fit=True)
        assert X_scaled.shape == X.shape
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)

    def test_scale_features_invalid_method(self):
        fe = FeatureEngineer()
        with pytest.raises(ValueError):
            fe.scale_features(np.ones((5, 3)), method="bad")

    def test_select_features_reduces_columns(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 20)
        y = (X[:, 0] > 0).astype(int)
        fe = FeatureEngineer()
        X_sel, indices = fe.select_features(X, y, method="f_classif", n_features=5)
        assert X_sel.shape[1] == 5
        assert len(indices) == 5

    def test_reduce_dimensions(self):
        X = np.random.RandomState(1).randn(50, 20)
        fe = FeatureEngineer()
        X_red = fe.reduce_dimensions(X, method="pca", n_components=5)
        assert X_red.shape == (50, 5)


class TestInteractionFeatures:
    def test_creates_interaction_columns(self):
        X = np.array([[1, 2], [3, 4]])
        names = ["a", "b"]
        X_int, new_names = create_interaction_features(X, names, interactions=[(0, 1)])
        assert X_int.shape[1] == 3
        assert new_names[-1] == "a_x_b"


class TestPolynomialFeatures:
    def test_creates_polynomial_columns(self):
        X = np.array([[2, 3], [4, 5]])
        names = ["a", "b"]
        X_poly, new_names = create_polynomial_features(X, names, degree=2)
        assert X_poly.shape[1] == 4  # original 2 + 2 squared
        assert "a^2" in new_names
