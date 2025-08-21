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

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# ---- Core imports for public API (must exist). If these fail, log full traceback. ----
try:
    from .main_service import (
        PortfolioSimulatorService,
        create_portfolio_simulator_service,
        simulate_portfolio_workflow,   # modular entrypoint
        get_simulation_charts,
    )
except Exception as e:
    # Make the root cause visible in server logs instead of a silent ImportError
    log.exception("Failed importing core simulator API from main_service")
    raise

# ---- Config (optional at import time; guard to avoid hard import failures) ----
try:
    from .config import (
        initialize_config,
        get_config,
        get_stock_metadata,
        get_risk_profiles,
    )
    _CONFIG_IMPORTED = True
except Exception as e:
    log.warning("Config module import warning: %s", e)
    initialize_config = None
    get_config = None
    get_stock_metadata = None
    get_risk_profiles = None
    _CONFIG_IMPORTED = False

# ---- Exceptions (safe to import; if these fail, still log and proceed) ----
try:
    from .exceptions import (
        PortfolioSimulatorError,
        ValidationError,
        DataProviderError,
        AIServiceError,
        VisualizationError,
        SimulationError,
        DatabaseError,
        SecurityError,
    )
except Exception as e:
    log.exception("Failed importing exceptions")
    raise

# ---- Components (non-critical at import time, but we expect them to exist) ----
try:
    from .validators import InputValidator
    from .data_provider import MarketDataProvider, DataQualityAnalyzer
    from .portfolio_simulator import PortfolioSimulator, PortfolioWeightCalculator
    from .ai_recommendation_service import AIRecommendationService, SHAPDataProcessor
    from .visualization_service import VisualizationService, ChartDataGenerator
    from .database_service import DatabaseService, SimulationResultsFormatter
except Exception as e:
    log.exception("Failed importing component modules")
    raise

# ---- Public API exports (modular-only; no legacy simulate_portfolio here) ----
__all__ = [
    # Main service classes
    "PortfolioSimulatorService",
    "create_portfolio_simulator_service",

    # Convenience functions (modular entrypoints)
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

# ---- Helpers to keep import-time initialization SAFE ----
def _safe(callable_fn, default):
    try:
        return callable_fn()
    except Exception as e:
        log.warning("Portfolio simulator init/metadata warning: %s", e)
        return default


# ---- Package-level convenience functions ----
async def quick_simulation(
    target_value: float,
    timeframe_years: int,
    risk_score: int,
    lump_sum: float = 0,
    monthly_contribution: float = 0,
    goal: str = "wealth building",
    db_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Quick portfolio simulation with minimal setup (modular workflow).
    """
    simulation_input = {
        "target_value": target_value,
        "timeframe": timeframe_years,
        "risk_score": risk_score,
        "risk_label": _map_risk_score_to_label(risk_score),
        "lump_sum": lump_sum,
        "monthly": monthly_contribution,
        "goal": goal,
        "years_of_experience": 0,
        "income_bracket": "medium",
    }

    if db_session is not None:
        # Full simulation with DB persistence via modular workflow
        return await simulate_portfolio_workflow(simulation_input, db_session)

    # In-memory simulation via service (DB ops are no-ops in the mock)
    service = create_portfolio_simulator_service()

    class _MockDB:
        def add(self, obj): ...
        def commit(self): ...
        def refresh(self, obj):
            obj.id = 999_999
            return obj
        def rollback(self): ...

    return await service.simulate_portfolio(simulation_input, _MockDB())


def validate_input(simulation_input: Dict[str, Any]) -> Dict[str, Any]:
    """Validate simulation input without running the simulation."""
    validator = InputValidator()
    return validator.validate_simulation_input(simulation_input)


def get_supported_stocks() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all supported stocks and ETFs."""
    if get_stock_metadata is None:
        return {}
    return _safe(lambda: get_stock_metadata(), {})


def assess_risk_profile(risk_score: int) -> Dict[str, Any]:
    """Get detailed risk profile info for a given risk score."""
    if get_risk_profiles is None:
        return {
            "risk_score": risk_score,
            "profile_name": _map_risk_score_to_profile(risk_score).title(),
            "risk_label": _map_risk_score_to_label(risk_score),
            "description": "Unavailable (risk profiles not loaded)",
        }

    risk_profiles = _safe(lambda: get_risk_profiles(), {})
    profile_key = _map_risk_score_to_profile(risk_score)
    profile_info = (risk_profiles.get(profile_key) or {}).copy()
    profile_info.update({
        "risk_score": risk_score,
        "profile_name": profile_key.title(),
        "risk_label": _map_risk_score_to_label(risk_score),
    })
    return profile_info


def estimate_returns(
    monthly_contribution: float,
    timeframe_years: int,
    expected_annual_return: float = 0.08,
) -> Dict[str, float]:
    """Simple compound return estimation."""
    monthly_rate = expected_annual_return / 12
    total_months = timeframe_years * 12

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
        "growth_percentage": round(
            (total_growth / total_contributions) * 100, 1
        ) if total_contributions > 0 else 0,
        "assumptions": {"annual_return": expected_annual_return, "compounding": "monthly"},
    }


# ---- Private helpers ----
def _map_risk_score_to_label(risk_score: int) -> str:
    if risk_score < 35:
        return "Conservative"
    elif risk_score < 70:
        return "Moderate"
    return "Aggressive"


def _map_risk_score_to_profile(risk_score: int) -> str:
    if risk_score < 35:
        return "conservative"
    elif risk_score < 70:
        return "moderate"
    return "aggressive"


# ---- Package initialization (SAFE) ----
def _initialize_package():
    """Initialize config without breaking imports if env/files are missing."""
    if initialize_config is None:
        return
    try:
        initialize_config()
        log.info("Portfolio Simulator package initialized successfully")
    except Exception as e:
        # Do NOT fail the package import if config initialization fails
        log.warning("Package initialization warning: %s", e)


_initialize_package()


# ---- Package metadata for introspection (SAFE) ----
_supported_assets = []
_risk_profiles = []

if get_stock_metadata is not None:
    _supported_assets = _safe(lambda: list(get_stock_metadata().keys()), [])

if get_risk_profiles is not None:
    _risk_profiles = _safe(lambda: list(get_risk_profiles().keys()), [])

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
        "Production-ready error handling",
    ],
    "supported_assets": _supported_assets,
    "risk_profiles": _risk_profiles,
}