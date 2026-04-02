"""
Data loading and preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_genomic_data(feature_path, label_path=None, test_size=0.2, random_state=42):
    """
    Load genomic feature matrix and labels.
    
    Parameters
    ----------
    feature_path : str
        Path to feature matrix (CSV)
    label_path : str, optional
        Path to labels (CSV)
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
    
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Training and test sets
    feature_names : list
        Feature names
    """
    # Load features
    X = pd.read_csv(feature_path, index_col=0)
    feature_names = X.columns.tolist()
    X = X.values
    
    # Load labels if provided
    if label_path:
        y_df = pd.read_csv(label_path, index_col=0)
        y = y_df.iloc[:, 0].values
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = None
    
    # Split data
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, feature_names
    else:
        return X, feature_names


def handle_missing_values(X, method='mean'):
    """
    Handle missing values in feature matrix.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    method : str
        'mean', 'median', or 'drop'
    
    Returns
    -------
    X_clean : np.ndarray
        Cleaned feature matrix
    """
    if isinstance(X, pd.DataFrame):
        if method == 'mean':
            X_clean = X.fillna(X.mean())
        elif method == 'median':
            X_clean = X.fillna(X.median())
        elif method == 'drop':
            X_clean = X.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        return X_clean.values
    else:
        # NumPy array
        if method == 'mean':
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
        elif method == 'median':
            col_median = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_median, inds[1])
        elif method == 'drop':
            X = X[~np.isnan(X).any(axis=1)]
        else:
            raise ValueError(f"Unknown method: {method}")
        return X


def remove_constant_features(X, feature_names=None):
    """
    Remove features with zero variance.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    feature_names : list, optional
        Feature names
    
    Returns
    -------
    X_filtered : np.ndarray
        Filtered feature matrix
    feature_names_filtered : list
        Filtered feature names
    """
    variances = np.var(X, axis=0)
    mask = variances > 0
    
    X_filtered = X[:, mask] if isinstance(X, np.ndarray) else X.iloc[:, mask].values
    
    if feature_names:
        feature_names_filtered = [name for name, keep in zip(feature_names, mask) if keep]
        return X_filtered, feature_names_filtered
    else:
        return X_filtered


def remove_highly_correlated_features(X, threshold=0.95, feature_names=None):
    """
    Remove highly correlated features.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix
    threshold : float
        Correlation threshold
    feature_names : list, optional
        Feature names
    
    Returns
    -------
    X_filtered : np.ndarray
        Filtered feature matrix
    feature_names_filtered : list
        Filtered feature names
    """
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    
    # Find highly correlated pairs
    to_drop = set()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)
    
    # Keep features not in to_drop
    mask = np.array([i not in to_drop for i in range(X.shape[1])])
    X_filtered = X[:, mask] if isinstance(X, np.ndarray) else X.iloc[:, mask].values
    
    if feature_names:
        feature_names_filtered = [name for name, keep in zip(feature_names, mask) if keep]
        return X_filtered, feature_names_filtered
    else:
        return X_filtered


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.
    """
    
    def __init__(self):
        self.means_ = None
        self.stds_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Fit preprocessor on training data by computing feature means and stds.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.means_ = np.nanmean(X, axis=0)
        self.stds_ = np.nanstd(X, axis=0)
        # Avoid division by zero: replace zero stds with 1.0
        self.stds_[self.stds_ == 0] = 1.0
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform data using stored means and stds (standard scaling).

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : np.ndarray
            Scaled feature matrix
        """
        if not self.is_fitted_:
            raise RuntimeError("DataPreprocessor has not been fitted. Call fit() first.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = (X - self.means_) / self.stds_
        return X_scaled

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
