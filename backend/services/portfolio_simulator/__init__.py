"""
Portfolio Simulator Service Package

A comprehensive, production-ready portfolio simulation service with:
- AI-powered stock recommendations
- SHAP explainable AI
- Advanced portfolio optimization
- Interactive visualizations
- Comprehensive error handling
- Security-focused input validation

This package provides both high-level convenience functions and detailed
component access for advanced use cases.
"""

__version__ = "2.0.0"
__author__ = "Portfolio Simulator Team"
__email__ = "support@portfoliosimulator.com"

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

# Configure logging for the package
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Core imports for public API
from .main_service import (
    PortfolioSimulatorService,
    create_portfolio_simulator_service,
    simulate_portfolio_workflow,
    get_simulation_charts
)

from .config import (
    initialize_config,
    get_config,
    get_stock_metadata,
    get_risk_profiles
)

from .exceptions import (
    PortfolioSimulatorError,
    ValidationError,
    DataProviderError,
    AIServiceError,
    VisualizationError,
    SimulationError,
    DatabaseError,
    SecurityError
)

# Component imports for advanced usage
from .validators import InputValidator
from .data_provider import MarketDataProvider, DataQualityAnalyzer
from .portfolio_simulator import PortfolioSimulator, PortfolioWeightCalculator
from .ai_recommendation_service import AIRecommendationService, SHAPDataProcessor
from .visualization_service import VisualizationService, ChartDataGenerator
from .database_service import DatabaseService, SimulationResultsFormatter

# Public API exports
__all__ = [
    # Main service classes
    "PortfolioSimulatorService",
    "create_portfolio_simulator_service",
    
    # Convenience functions
    "simulate_portfolio_workflow",
    "get_simulation_charts",
    
    # Configuration
    "initialize_config",
    "get_config",
    "get_stock_metadata",
    "get_risk_profiles",
    
    # Exceptions
    "PortfolioSimulatorError",
    "ValidationError",
    "DataProviderError",
    "AIServiceError",
    "VisualizationError",
    "SimulationError",
    "DatabaseError",
    "SecurityError",
    
    # Component classes
    "InputValidator",
    "MarketDataProvider",
    "DataQualityAnalyzer",
    "PortfolioSimulator",
    "PortfolioWeightCalculator",
    "AIRecommendationService",
    "SHAPDataProcessor",
    "VisualizationService",
    "ChartDataGenerator",
    "DatabaseService",
    "SimulationResultsFormatter",
]


# Package-level convenience functions
async def quick_simulation(
    target_value: float,
    timeframe_years: int,
    risk_score: int,
    lump_sum: float = 0,
    monthly_contribution: float = 0,
    goal: str = "wealth building",
    db_session: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Quick portfolio simulation with minimal setup.
    
    This is a convenience function for simple simulations without
    requiring detailed input preparation.
    
    Args:
        target_value: Investment target amount (£)
        timeframe_years: Investment period in years
        risk_score: Risk tolerance (0-100)
        lump_sum: Initial investment amount (£)
        monthly_contribution: Monthly investment amount (£)
        goal: Investment goal description
        db_session: Optional database session for persistence
        
    Returns:
        Complete simulation results
        
    Example:
        >>> import portfolio_simulator as ps
        >>> result = await ps.quick_simulation(
        ...     target_value=100000,
        ...     timeframe_years=10,
        ...     risk_score=60,
        ...     lump_sum=10000,
        ...     monthly_contribution=500
        ... )
        >>> print(f"Final value: £{result['results']['end_value']:,.2f}")
    """
    # Prepare simulation input
    simulation_input = {
        "target_value": target_value,
        "timeframe": timeframe_years,
        "risk_score": risk_score,
        "risk_label": _map_risk_score_to_label(risk_score),
        "lump_sum": lump_sum,
        "monthly": monthly_contribution,
        "goal": goal,
        "years_of_experience": 0,
        "income_bracket": "medium"
    }
    
    if db_session:
        # Full simulation with database persistence
        return await simulate_portfolio_workflow(simulation_input, db_session)
    else:
        # In-memory simulation only
        service = create_portfolio_simulator_service()
        
        # Create a mock database session for validation
        # In a real implementation, you'd need to handle this properly
        class MockDB:
            def add(self, obj): pass
            def commit(self): pass
            def refresh(self, obj): 
                obj.id = 999999  # Mock ID
                return obj
            def rollback(self): pass
        
        mock_db = MockDB()
        return await service.simulate_portfolio(simulation_input, mock_db)


def validate_input(simulation_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate simulation input without running simulation.
    
    Args:
        simulation_input: Raw input data
        
    Returns:
        Validated input data
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> import portfolio_simulator as ps
        >>> try:
        ...     validated = ps.validate_input({
        ...         "target_value": 50000,
        ...         "timeframe": 10,
        ...         "risk_score": 75
        ...     })
        ...     print("Input is valid!")
        ... except ps.ValidationError as e:
        ...     print(f"Validation failed: {e.message}")
    """
    validator = InputValidator()
    return validator.validate_simulation_input(simulation_input)


def get_supported_stocks() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all supported stocks and ETFs.
    
    Returns:
        Dictionary mapping ticker symbols to metadata
        
    Example:
        >>> import portfolio_simulator as ps
        >>> stocks = ps.get_supported_stocks()
        >>> print(f"VTI: {stocks['VTI']['name']}")
        VTI: Vanguard Total Stock Market ETF
    """
    return get_stock_metadata()


def assess_risk_profile(risk_score: int) -> Dict[str, Any]:
    """
    Get detailed risk profile information for a given risk score.
    
    Args:
        risk_score: Risk tolerance score (0-100)
        
    Returns:
        Risk profile information
        
    Example:
        >>> import portfolio_simulator as ps
        >>> profile = ps.assess_risk_profile(45)
        >>> print(profile['description'])
        Balanced growth with moderate risk
    """
    risk_profiles = get_risk_profiles()
    profile_key = _map_risk_score_to_profile(risk_score)
    
    profile_info = risk_profiles[profile_key].copy()
    profile_info.update({
        "risk_score": risk_score,
        "profile_name": profile_key.title(),
        "risk_label": _map_risk_score_to_label(risk_score)
    })
    
    return profile_info


def estimate_returns(
    monthly_contribution: float,
    timeframe_years: int,
    expected_annual_return: float = 0.08
) -> Dict[str, float]:
    """
    Simple compound return estimation.
    
    Args:
        monthly_contribution: Monthly investment amount
        timeframe_years: Investment period in years
        expected_annual_return: Expected annual return (default 8%)
        
    Returns:
        Dictionary with return estimates
        
    Example:
        >>> import portfolio_simulator as ps
        >>> estimates = ps.estimate_returns(500, 10, 0.09)
        >>> print(f"Final value: £{estimates['final_value']:,.2f}")
    """
    monthly_rate = expected_annual_return / 12
    total_months = timeframe_years * 12
    
    # Future value of annuity formula
    if monthly_rate > 0:
        final_value = monthly_contribution * (
            ((1 + monthly_rate) ** total_months - 1) / monthly_rate
        )
    else:
        final_value = monthly_contribution * total_months
    
    total_contributions = monthly_contribution * total_months
    total_growth = final_value - total_contributions
    
    return {
        "final_value": round(final_value, 2),
        "total_contributions": round(total_contributions, 2),
        "total_growth": round(total_growth, 2),
        "growth_percentage": round((total_growth / total_contributions) * 100, 1) if total_contributions > 0 else 0,
        "assumptions": {
            "annual_return": expected_annual_return,
            "compounding": "monthly"
        }
    }


# Private helper functions
def _map_risk_score_to_label(risk_score: int) -> str:
    """Map risk score to human-readable label."""
    if risk_score < 35:
        return "Conservative"
    elif risk_score < 70:
        return "Moderate"
    else:
        return "Aggressive"


def _map_risk_score_to_profile(risk_score: int) -> str:
    """Map risk score to profile key."""
    if risk_score < 35:
        return "conservative"
    elif risk_score < 70:
        return "moderate"
    else:
        return "aggressive"


# Package initialization
def _initialize_package():
    """Initialize the package with default configuration."""
    try:
        # Attempt to initialize configuration
        initialize_config()
        logging.getLogger(__name__).info("Portfolio Simulator package initialized successfully")
    except Exception as e:
        # Don't fail package import if config initialization fails
        logging.getLogger(__name__).warning(f"Package initialization warning: {e}")


# Initialize package on import
_initialize_package()


# Package metadata for introspection
PACKAGE_INFO = {
    "name": "portfolio_simulator",
    "version": __version__,
    "description": "AI-powered portfolio simulation with explainable recommendations",
    "features": [
        "AI stock recommendations",
        "SHAP explainable AI",
        "Portfolio optimization",
        "Interactive visualizations",
        "Comprehensive validation",
        "Production-ready error handling"
    ],
    "supported_assets": list(get_stock_metadata().keys()),
    "risk_profiles": list(get_risk_profiles().keys())
}