"""
Analysis Module for WealthWise Enhanced Stock Recommender

This module contains the core analysis components that power the AI-driven
investment recommendation system:

1. MarketRegimeDetector - Analyzes current market conditions and trends
2. FactorAnalyzer - Multi-factor quantitative analysis for stock evaluation  
3. PortfolioOptimizer - Advanced portfolio construction and optimization

These components work together to provide sophisticated, institutional-quality
investment analysis while maintaining transparency and educational value.
"""

from .market_regime import MarketRegimeDetector
from .factor_analysis import FactorAnalyzer
from .portfolio_optimizer import PortfolioOptimizer

__all__ = [
    'MarketRegimeDetector',
    'FactorAnalyzer', 
    'PortfolioOptimizer'
]

# Version info
__version__ = '1.0.0'
__author__ = 'WealthWise Team'
__description__ = 'AI-powered investment analysis and portfolio optimization'