"""
ML Classification for Genomic Data
"""

from .data import load_genomic_data, DataPreprocessor
from .features import FeatureEngineer
from .neural_net import BiologicalClassifier, ConvolutionalClassifier, ModelTrainer
from .evaluation import ModelEvaluator, compare_models

__version__ = "1.0.0"
__author__ = "Adam Hoffman"

__all__ = [
    'load_genomic_data',
    'DataPreprocessor',
    'FeatureEngineer',
    'BiologicalClassifier',
    'ConvolutionalClassifier',
    'ModelTrainer',
    'ModelEvaluator',
    'compare_models'
]
