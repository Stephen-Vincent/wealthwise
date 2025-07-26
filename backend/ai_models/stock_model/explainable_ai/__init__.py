"""
Explainable AI Module for WealthWise Enhanced Stock Recommender

This module implements SHAP (SHapley Additive exPlanations) and advanced
visualization techniques to make AI investment decisions transparent and
educational. The goal is to ensure users understand not just WHAT investments
are recommended, but WHY they are recommended.

Core Components:
1. SHAPExplainer - SHAP-powered explanations for portfolio recommendations
2. VisualizationEngine - Professional charts and interactive dashboards

Key Features:
- Transparent ML decision-making using SHAP values
- Human-readable explanations for every recommendation factor
- Professional-quality visualizations and charts
- Interactive dashboards for exploring AI decisions
- Educational transparency to build user trust and understanding

Dependencies:
- Required: sklearn, numpy, pandas, matplotlib, seaborn
- Optional: shap (for explainable AI), plotly (for interactive charts)

Example Usage:
    from ai_models.stock_model.explainable_ai import SHAPExplainer, VisualizationEngine
    
    # Initialize explainer
    explainer = SHAPExplainer()
    explainer.train_shap_model()
    
    # Get explanation for a recommendation
    explanation = explainer.get_shap_explanation(
        target_value=50000, timeframe=10, risk_score=65
    )
    
    # Create visualizations
    viz_engine = VisualizationEngine()
    viz_engine.create_shap_waterfall_chart(explanation)
"""

from .shap_explainer import SHAPExplainer
from .visualization import VisualizationEngine

# Check for optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

__all__ = [
    'SHAPExplainer',
    'VisualizationEngine',
    'SHAP_AVAILABLE',
    'PLOTLY_AVAILABLE'
]

# Version and metadata
__version__ = '1.0.0'
__author__ = 'WealthWise Team'
__description__ = 'Explainable AI and visualization tools for transparent investment recommendations'

# Dependency warnings
if not SHAP_AVAILABLE:
    import warnings
    warnings.warn(
        "SHAP not available. Install with 'pip install shap' for explainable AI features.",
        ImportWarning
    )

if not PLOTLY_AVAILABLE:
    import warnings
    warnings.warn(
        "Plotly not available. Install with 'pip install plotly' for interactive visualizations.",
        ImportWarning
    )

def check_dependencies() -> dict:
    """
    Check availability of optional dependencies
    
    Returns:
        dict: Status of each optional dependency
    """
    return {
        'shap': SHAP_AVAILABLE,
        'plotly': PLOTLY_AVAILABLE,
        'core_features': True  # Core matplotlib/seaborn always available
    }

def get_installation_instructions() -> str:
    """
    Get installation instructions for missing dependencies
    
    Returns:
        str: Installation commands for missing packages
    """
    missing = []
    
    if not SHAP_AVAILABLE:
        missing.append("pip install shap")
    
    if not PLOTLY_AVAILABLE:
        missing.append("pip install plotly")
    
    if missing:
        return "Install missing dependencies:\n" + "\n".join(missing)
    else:
        return "All optional dependencies are available!"