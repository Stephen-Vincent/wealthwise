"""
Portfolio Simulator Modules Package

Contains all the modular components for the enhanced portfolio simulator:
- Goal Calculator: Smart return calculations
- Stock Recommender: Enhanced AI recommendations  
- Data Manager: Robust stock data handling
- Portfolio Optimizer: Advanced optimization algorithms
- Simulation Engine: Enhanced simulation with debugging
- Crash Analyzer: Market crash detection with news
- AI Summarizer: Enhanced summaries with SHAP
- Database Manager: Robust database operations
- Serialization Utils: Comprehensive JSON handling
"""

# Import all modules for easy access
from .goal_calculator import SmartGoalCalculator
from .stock_recommender import EnhancedStockRecommender
from .data_manager import StockDataManager
from .portfolio_optimizer import PortfolioOptimizer
from .simulation_engine import SimulationEngine
from .crash_analyzer import MarketCrashAnalyzer
from .ai_summarizer import AISummaryGenerator
from .database_manager import DatabaseManager, SerializationManager
from .serialization_utils import SerializationManager as SerializationUtils

__all__ = [
    "SmartGoalCalculator",
    "EnhancedStockRecommender", 
    "StockDataManager",
    "PortfolioOptimizer",
    "SimulationEngine",
    "MarketCrashAnalyzer",
    "AISummaryGenerator",
    "DatabaseManager",
    "SerializationManager",
    "SerializationUtils"
]