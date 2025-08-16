"""
Enhanced Portfolio Simulator Package

This package provides a modular, enhanced portfolio simulation system with:
- Smart goal calculations (fixes 0% return issue)
- Market crash detection and news analysis
- SHAP explainable AI integration
- Robust error handling and fallbacks
- Comprehensive serialization support
"""

from .main import simulate_portfolio, get_simulation_crash_analysis, generate_shap_visualization

__version__ = "1.0.0"
__author__ = "WealthWise Team"

# Make main functions available at package level
__all__ = [
    "simulate_portfolio",
    "get_simulation_crash_analysis", 
    "generate_shap_visualization"
]