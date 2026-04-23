"""
End-to-end integration test: synthetic-GDSC -> preprocess -> train -> evaluate.

Ensures the pipeline actually runs on a fresh clone (no external data) and
that the classifiers recover signal on the planted driver features.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = PROJECT_ROOT / "data" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_gdsc():
    """Ensure the synthetic fixture exists; generate it if missing."""
    if not (SYN_DIR / "features.csv").is_file():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_synthetic_gdsc.py")],
            check=True,
        )
    assert (SYN_DIR / "features.csv").is_file()
    assert (SYN_DIR / "labels.csv").is_file()
    return SYN_DIR


def test_load_and_split(synthetic_gdsc):
    from src.data import load_genomic_data

    X_train, X_test, y_train, y_test, feature_names = load_genomic_data(
        synthetic_gdsc / "features.csv",
        synthetic_gdsc / "labels.csv",
        test_size=0.25,
        random_state=0,
    )
    assert X_train.shape[0] + X_test.shape[0] == 200
    assert X_train.shape[1] == 50
    assert len(feature_names) == 50
    assert set(np.unique(y_train)) <= {0, 1}


def test_preprocessor_roundtrip(synthetic_gdsc):
    from src.data import load_genomic_data, DataPreprocessor

    X_train, X_test, y_train, y_test, _ = load_genomic_data(
        synthetic_gdsc / "features.csv",
        synthetic_gdsc / "labels.csv",
        test_size=0.25,
        random_state=0,
    )
    pp = DataPreprocessor()
    X_train_scaled = pp.fit_transform(X_train)
    X_test_scaled = pp.transform(X_test)
    # train-fold should be centred
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-10)
    # test-fold is transformed by train stats -> shape preserved, not necessarily centred
    assert X_test_scaled.shape == X_test.shape


@pytest.mark.parametrize("trainer_name", [
    "train_logistic_regression",
    "train_random_forest",
    "train_gradient_boosting",
])
def test_model_recovers_signal(synthetic_gdsc, trainer_name):
    """
    Classifiers with non-trivial capacity should beat chance on the planted
    driver features (AUC comfortably > 0.6 on 5-fold CV).
    """
    from src import models as mdl
    from src.data import load_genomic_data, DataPreprocessor

    X_train, X_test, y_train, y_test, _ = load_genomic_data(
        synthetic_gdsc / "features.csv",
        synthetic_gdsc / "labels.csv",
        test_size=0.25,
        random_state=0,
    )
    pp = DataPreprocessor()
    X_train_scaled = pp.fit_transform(X_train)
    X_test_scaled = pp.transform(X_test)

    trainer = getattr(mdl, trainer_name)
    model, metrics = trainer(X_train_scaled, y_train, cv_folds=5)

    # CV AUC should reflect real signal on the driver features.
    assert metrics["roc_auc_mean"] > 0.6, (
        f"{trainer_name} CV AUC={metrics['roc_auc_mean']:.3f} — below 0.6, "
        "signal not recovered from synthetic driver features."
    )

    held_out = mdl.evaluate_model(model, X_test_scaled, y_test)
    assert 0.0 <= held_out["accuracy"] <= 1.0
    # AUC may be NaN if y_test is single-class; tolerate that.
    assert "auc" in held_out


def test_compare_models_shape(synthetic_gdsc):
    from src import models as mdl
    from src.data import load_genomic_data, DataPreprocessor

    X_train, X_test, y_train, y_test, _ = load_genomic_data(
        synthetic_gdsc / "features.csv",
        synthetic_gdsc / "labels.csv",
        test_size=0.25,
        random_state=0,
    )
    pp = DataPreprocessor()
    X_train_scaled = pp.fit_transform(X_train)
    X_test_scaled = pp.transform(X_test)

    results = {}
    for name, trainer in [
        ("lr", mdl.train_logistic_regression),
        ("rf", mdl.train_random_forest),
    ]:
        model, _ = trainer(X_train_scaled, y_train, cv_folds=3)
        results[name] = mdl.evaluate_model(model, X_test_scaled, y_test)
    summary = mdl.compare_models(results)
    assert summary.shape[0] == 2
    assert "accuracy" in summary.columns
