"""
ML Classification for Genomic Data
"""

from .data import DataPreprocessor, load_genomic_data
from .evaluation import ModelEvaluator, compare_models
from .features import FeatureEngineer
from .neural_net import BiologicalClassifier, ConvolutionalClassifier, ModelTrainer

__version__ = "1.0.0"
__author__ = "Adam Hoffman"

__all__ = [
    "load_genomic_data",
    "DataPreprocessor",
    "FeatureEngineer",
    "BiologicalClassifier",
    "ConvolutionalClassifier",
    "ModelTrainer",
    "ModelEvaluator",
    "compare_models",
]
