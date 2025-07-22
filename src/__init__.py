"""
Aave V2 Credit Scoring Package.

This package provides modules for feature engineering and credit scoring based on
Aave V2 transaction data.

By importing the key components here, they can be accessed directly from the `src` package,
simplifying imports in other parts of the application.
"""

from .feature_engineering import engineer_features
from .scoring_model import CreditScoringModel

__version__ = "1.0.0"
__author__ = "Sanjay Murugadoss"

print("Initializing 'src' credit scoring package...")