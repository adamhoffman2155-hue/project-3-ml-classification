"""
Tests for data loading and preprocessing.
"""

import numpy as np
import pytest

from src.data import handle_missing_values, remove_constant_features, DataPreprocessor


class TestHandleMissingValues:
    def test_mean_imputation(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        result = handle_missing_values(X, method="mean")
        assert not np.isnan(result).any()
        assert result.shape == (3, 2)

    def test_median_imputation(self):
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        result = handle_missing_values(X, method="median")
        assert not np.isnan(result).any()

    def test_invalid_method_raises(self):
        X = np.array([[1.0]])
        with pytest.raises(ValueError):
            handle_missing_values(X, method="invalid")


class TestRemoveConstantFeatures:
    def test_removes_constant_column(self):
        X = np.array([[1, 2, 3], [1, 5, 6], [1, 8, 9]])
        result = remove_constant_features(X)
        assert result.shape[1] == 2


class TestDataPreprocessor:
    def test_fit_stores_statistics(self):
        pp = DataPreprocessor()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pp.fit(X)
        assert pp.means_ is not None
        assert pp.stds_ is not None
        assert pp.is_fitted_

    def test_transform_scales_data(self):
        pp = DataPreprocessor()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_scaled = pp.fit_transform(X)
        # After standard scaling, mean should be ~0 and std ~1
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)

    def test_transform_before_fit_raises(self):
        pp = DataPreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(np.array([[1.0, 2.0]]))
