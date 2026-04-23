"""
ML Classification for Genomic Data
"""

from .data import load_genomic_data, DataPreprocessor
from .features import FeatureEngineer
from .evaluation import ModelEvaluator, compare_models

# neural_net requires torch which is a heavy optional dep. Import it lazily
# via `from src.neural_net import ...` rather than at package init so the rest
# of the package works in torch-free environments.

__version__ = "1.0.0"
__author__ = "Adam Hoffman"

__all__ = [
    'load_genomic_data',
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelEvaluator',
    'compare_models'
]
