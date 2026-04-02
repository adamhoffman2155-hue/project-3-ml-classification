"""
Classical ML model wrappers for genomic classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer,
)


def _run_stratified_cv(model, X, y, cv_folds=5):
    """
    Run stratified cross-validation and return model + metrics.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    model : estimator
        Fitted model (on full training data).
    metrics : dict
        Mean and std of each metric across folds.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1": make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc": make_scorer(
            roc_auc_score,
            multi_class="ovr",
            needs_proba=True,
            average="weighted",
        ),
    }

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=False
    )

    metrics = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        metrics[f"{metric_name}_mean"] = float(np.mean(cv_results[key]))
        metrics[f"{metric_name}_std"] = float(np.std(cv_results[key]))

    # Fit on full training set
    model.fit(X, y)
    return model, metrics


def train_logistic_regression(X, y, cv_folds=5):
    """
    Train a logistic regression classifier with stratified CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    model : LogisticRegression
        Fitted model.
    metrics : dict
        Cross-validation metrics.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
        n_jobs=-1,
    )
    return _run_stratified_cv(model, X, y, cv_folds)


def train_random_forest(X, y, cv_folds=5):
    """
    Train a random forest classifier with stratified CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    model : RandomForestClassifier
        Fitted model.
    metrics : dict
        Cross-validation metrics.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return _run_stratified_cv(model, X, y, cv_folds)


def train_gradient_boosting(X, y, cv_folds=5):
    """
    Train a gradient boosting classifier with stratified CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    model : GradientBoostingClassifier
        Fitted model.
    metrics : dict
        Cross-validation metrics.
    """
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
    )
    return _run_stratified_cv(model, X, y, cv_folds)


def train_elastic_net(X, y, cv_folds=5):
    """
    Train an elastic net classifier (SGD with elastic net penalty) with stratified CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    cv_folds : int
        Number of CV folds.

    Returns
    -------
    model : SGDClassifier
        Fitted model.
    metrics : dict
        Cross-validation metrics.
    """
    model = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha=1e-4,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    return _run_stratified_cv(model, X, y, cv_folds)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a fitted model on a test set.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn estimator.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True labels.

    Returns
    -------
    metrics : dict
        Dictionary with AUC, accuracy, precision, recall, f1.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    # AUC requires probability estimates
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        try:
            metrics["auc"] = float(
                roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["auc"] = float("nan")
    elif hasattr(model, "decision_function"):
        y_decision = model.decision_function(X_test)
        try:
            metrics["auc"] = float(
                roc_auc_score(y_test, y_decision, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")

    return metrics


def compare_models(results_dict):
    """
    Create a summary DataFrame comparing multiple models.

    Parameters
    ----------
    results_dict : dict
        Mapping of model name -> metrics dict (as returned by evaluate_model
        or the CV metrics from train_* functions).

    Returns
    -------
    summary : pd.DataFrame
        Summary table with models as rows and metrics as columns,
        sorted by f1 descending.
    """
    rows = []
    for name, metrics in results_dict.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    # Sort by f1 (or f1_mean if from CV) descending
    sort_col = "f1_mean" if "f1_mean" in summary.columns else "f1"
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False)
    return summary
