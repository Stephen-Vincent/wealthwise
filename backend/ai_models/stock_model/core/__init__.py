# ai_models/stock_model/core/__init__.py

"""
Core components of the Enhanced Stock Recommender system.

This package contains the main recommendation engine and configuration.
"""

from .recommender import EnhancedStockRecommender
from .config import (
    ASSET_UNIVERSES,
    BACKUP_TICKERS,
    RELIABLE_TICKERS,
    RISK_SCORE_TO_CATEGORY,
    CATEGORY_TO_RISK_SCORE,
    SHAP_FEATURE_NAMES,
    DEFAULT_FACTOR_WEIGHTS,
    REGIME_ADJUSTMENTS,
    get_risk_category
)

__all__ = [
    "EnhancedStockRecommender",
    "ASSET_UNIVERSES",
    "BACKUP_TICKERS", 
    "RELIABLE_TICKERS",
    "RISK_SCORE_TO_CATEGORY",
    "CATEGORY_TO_RISK_SCORE",
    "SHAP_FEATURE_NAMES",
    "DEFAULT_FACTOR_WEIGHTS",
    "REGIME_ADJUSTMENTS",
    "get_risk_category"
]