"""
General utility functions.
"""

import logging
import os
import random

import numpy as np


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and (optionally) PyTorch.

    Parameters
    ----------
    seed : int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(name, level=logging.INFO):
    """
    Configure and return a logger with a standard format.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path):
    """
    Create directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
