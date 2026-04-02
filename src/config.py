"""
Configuration management for model hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Hyperparameters for each supported model."""

    # Logistic Regression
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"
    lr_C: float = 1.0

    # Random Forest
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2

    # Gradient Boosting
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 5
    gb_min_samples_split: int = 5
    gb_min_samples_leaf: int = 2
    gb_subsample: float = 0.8

    # Elastic Net (SGD)
    en_l1_ratio: float = 0.5
    en_alpha: float = 1e-4
    en_max_iter: int = 1000

    # General
    random_state: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    n_features: int = 500
    scaling_method: str = "standard"
    feature_selection_method: str = "mutual_info"


# Default configuration instance
DEFAULT_CONFIG = ModelConfig()


def load_config(yaml_path):
    """
    Load a ModelConfig from a YAML file.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.

    Returns
    -------
    config : ModelConfig
        Populated configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Only pass keys that are valid ModelConfig fields
    valid_fields = {fld.name for fld in ModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}

    return ModelConfig(**filtered)
