"""
Feature engineering and selection utilities
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, chi2, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier


class FeatureEngineer:
    """
    Feature engineering and selection pipeline.
    """
    
    def __init__(self):
        self.scaler = None
        self.selector = None
        self.pca = None
    
    def scale_features(self, X, method='standard', fit=True):
        """
        Scale features using specified method.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        method : str
            'standard', 'minmax', or 'robust'
        fit : bool
            Whether to fit scaler
        
        Returns
        -------
        X_scaled : np.ndarray
            Scaled feature matrix
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if fit:
            self.scaler = scaler
            X_scaled = scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def select_features(self, X, y, method='mutual_info', n_features=500, fit=True):
        """
        Select top features using specified method.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        method : str
            'mutual_info', 'f_classif', or 'random_forest'
        n_features : int
            Number of features to select
        fit : bool
            Whether to fit selector
        
        Returns
        -------
        X_selected : np.ndarray
            Selected feature matrix
        selected_indices : np.ndarray
            Indices of selected features
        """
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        elif method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            selector = SelectFromModel(rf, prefit=False, max_features=n_features)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        if fit:
            self.selector = selector
            X_selected = selector.fit_transform(X, y)
        else:
            if self.selector is None:
                raise ValueError("Selector not fitted. Call with fit=True first.")
            X_selected = selector.transform(X)
        
        # Get selected feature indices
        if hasattr(selector, 'get_support'):
            selected_indices = np.where(selector.get_support())[0]
        else:
            selected_indices = np.arange(X_selected.shape[1])
        
        return X_selected, selected_indices
    
    def reduce_dimensions(self, X, method='pca', n_components=50, fit=True):
        """
        Reduce dimensionality using specified method.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        method : str
            'pca' or 'other'
        n_components : int
            Number of components
        fit : bool
            Whether to fit reducer
        
        Returns
        -------
        X_reduced : np.ndarray
            Reduced feature matrix
        """
        if method == 'pca':
            reducer = PCA(n_components=min(n_components, X.shape[1]))
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        if fit:
            self.pca = reducer
            X_reduced = reducer.fit_transform(X)
        else:
            if self.pca is None:
                raise ValueError("Reducer not fitted. Call with fit=True first.")
            X_reduced = self.pca.transform(X)
        
        return X_reduced
    
    def get_feature_importance(self, feature_names, top_n=20):
        """
        Get top important features from selector.
        
        Parameters
        ----------
        feature_names : list
            Original feature names
        top_n : int
            Number of top features to return
        
        Returns
        -------
        important_features : list
            Top important feature names
        """
        if self.selector is None:
            raise ValueError("No selector fitted")
        
        if hasattr(self.selector, 'scores_'):
            scores = self.selector.scores_
            top_indices = np.argsort(scores)[-top_n:][::-1]
            important_features = [feature_names[i] for i in top_indices]
            return important_features
        else:
            raise ValueError("Selector does not have feature scores")


def create_interaction_features(X, feature_names, interactions=None):
    """
    Create interaction features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    interactions : list of tuples, optional
        Pairs of feature indices to interact
    
    Returns
    -------
    X_interactions : np.ndarray
        Feature matrix with interactions
    new_feature_names : list
        Updated feature names
    """
    X_interactions = X.copy()
    new_feature_names = feature_names.copy()
    
    if interactions is None:
        # Create all pairwise interactions (expensive!)
        interactions = [(i, j) for i in range(X.shape[1]) for j in range(i+1, X.shape[1])]
    
    for i, j in interactions:
        interaction = X[:, i] * X[:, j]
        X_interactions = np.column_stack([X_interactions, interaction])
        new_feature_names.append(f"{feature_names[i]}_x_{feature_names[j]}")
    
    return X_interactions, new_feature_names


def create_polynomial_features(X, feature_names, degree=2):
    """
    Create polynomial features.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list
        Feature names
    degree : int
        Polynomial degree
    
    Returns
    -------
    X_poly : np.ndarray
        Feature matrix with polynomial features
    new_feature_names : list
        Updated feature names
    """
    X_poly = X.copy()
    new_feature_names = feature_names.copy()
    
    for d in range(2, degree + 1):
        for i, name in enumerate(feature_names):
            poly_feature = X[:, i] ** d
            X_poly = np.column_stack([X_poly, poly_feature])
            new_feature_names.append(f"{name}^{d}")
    
    return X_poly, new_feature_names
