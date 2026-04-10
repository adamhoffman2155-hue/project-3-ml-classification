"""
Model evaluation and visualization utilities
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve,
)


class ModelEvaluator:
    """Comprehensive model evaluation and visualization."""

    def __init__(self, figsize=(10, 8), dpi=300):
        self.figsize = figsize
        self.dpi = dpi

    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None, label=None):
        """Plot ROC curve and return AUC."""
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=self.figsize)
        plt.plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
                 label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve" + (f" - {label}" if label else ""))
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return roc_auc

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix and return the matrix."""
        cm = confusion_matrix(y_true, y_pred)

        fig = plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"label": "Count"})
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return cm

    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot precision-recall curve."""
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        fig = plt.figure(figsize=self.figsize)
        plt.plot(recall, precision, color="blue", lw=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        """Plot feature importances for models exposing feature_importances_."""
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]

        fig = plt.figure(figsize=self.figsize)
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def plot_learning_curves(self, train_scores, val_scores, save_path=None):
        """Plot training/validation learning curves."""
        fig = plt.figure(figsize=self.figsize)
        plt.plot(train_scores, label="Training", marker="o")
        plt.plot(val_scores, label="Validation", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def generate_report(self, y_true, y_pred, class_names=None):
        """Return a sklearn classification report as a string."""
        return classification_report(y_true, y_pred, target_names=class_names)
