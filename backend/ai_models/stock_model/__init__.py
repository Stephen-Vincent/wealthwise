# ai_models/stock_model/__init__.py

"""
WealthWise Enhanced Stock Recommender Module

This module provides AI-powered, explainable investment recommendations
with goal-oriented optimization and transparent decision-making.

Main Components:
- EnhancedStockRecommender: Core recommendation engine
- SHAP Explainable AI: Transparent explanations for all recommendations
- Market Regime Detection: Adaptive recommendations based on market conditions
- Multi-Factor Analysis: Professional-grade quantitative stock evaluation

Usage:
    from ai_models.stock_model import EnhancedStockRecommender
    
    recommender = EnhancedStockRecommender()
    stocks = recommender.recommend_stocks(50000, 10, 65, 5000, 300)
    explanation = recommender.get_shap_explanation(50000, 10, 65, 5000, 300)
"""

from .core.recommender import EnhancedStockRecommender
from .core.config import (
    ASSET_UNIVERSES,
    BACKUP_TICKERS, 
    RELIABLE_TICKERS,
    get_risk_category
)

# Backward compatibility functions
def train_and_recommend(target_value: float, timeframe: int, risk_score: float):
    """Backward compatible function for existing code."""
    recommender = EnhancedStockRecommender()
    return recommender.recommend_stocks(target_value, timeframe, risk_score)

def get_recommendation_explanation(target_value: float, timeframe: int, risk_score: float,
                                 current_investment: float = 0, monthly_contribution: float = 0):
    """Get SHAP explanation for recommendations."""
    recommender = EnhancedStockRecommender()
    return recommender.get_shap_explanation(target_value, timeframe, risk_score, 
                                          current_investment, monthly_contribution)

def get_backup_tickers(count: int = 5):
    """Get reliable backup tickers."""
    return RELIABLE_TICKERS[:count]

__version__ = "2.0.0"
__author__ = "WealthWise AI Team"

__all__ = [
    "EnhancedStockRecommender",
    "train_and_recommend", 
    "get_recommendation_explanation",
    "get_backup_tickers",
    "ASSET_UNIVERSES",
    "BACKUP_TICKERS",
    "get_risk_category"
]