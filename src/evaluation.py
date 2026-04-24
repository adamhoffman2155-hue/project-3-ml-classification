"""
Model evaluation and visualization utilities
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization.
    """

    def __init__(self, figsize=(10, 8), dpi=300):
        """Initialize evaluator."""
        self.figsize = figsize
        self.dpi = dpi

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None, label=None):
        """
        Plot ROC curve.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        save_path : str, optional
            Path to save figure
        label : str, optional
            Model label
        """
        # Handle multi-class by taking positive class probability
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve" + (f" - {label}" if label else ""))
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        return roc_auc

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """
        Plot confusion matrix.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : list, optional
            Class names
        save_path : str, optional
            Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=self.figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Count"},
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

        return cm

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot precision-recall curve.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        save_path : str, optional
            Path to save figure
        """
        # Handle multi-class
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        plt.figure(figsize=self.figsize)
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance.

        Parameters
        ----------
        model : sklearn model
            Trained model with feature_importances_ attribute
        feature_names : list
            Feature names
        top_n : int
            Number of top features to plot
        save_path : str, optional
            Path to save figure
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]

        plt.figure(figsize=self.figsize)
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

    def plot_learning_curves(self, train_scores, val_scores, save_path=None):
        """
        Plot learning curves.

        Parameters
        ----------
        train_scores : list
            Training scores
        val_scores : list
            Validation scores
        save_path : str, optional
            Path to save figure
        """
        plt.figure(figsize=self.figsize)
        plt.plot(train_scores, label="Training", marker="o")
        plt.plot(val_scores, label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.show()

    def generate_report(self, y_true, y_pred, class_names=None):
        """
        Generate classification report.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : list, optional
            Class names

        Returns
        -------
        report : str
            Classification report
        """
        report = classification_report(y_true, y_pred, target_names=class_names)
        print(report)
        return report


def compare_models(models_dict, X_test, y_test):
    """
    Compare multiple models.

    Parameters
    ----------
    models_dict : dict
        Dictionary of {model_name: model}
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels

    Returns
    -------
    results_df : pd.DataFrame
        Comparison results
    """
    results = []

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)

        # Handle probability predictions
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            roc_auc = auc(*roc_curve(y_test, y_pred_proba[:, 1])[:2])
        else:
            roc_auc = np.nan

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        results.append({"Model": name, "Accuracy": accuracy, "F1-Score": f1, "ROC-AUC": roc_auc})

    results_df = pd.DataFrame(results)
    return results_df
