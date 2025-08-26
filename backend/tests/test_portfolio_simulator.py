# backend/tests/test_portfolio_simulator.py

import sys
import types
import pytest
import pandas as pd
import numpy as np

# --- Mock the `database` package early to satisfy `database_service` imports ---
# Your package __init__ imports main_service -> database_service -> `from database import models`
# We inject a tiny stand-in so the import succeeds without a real DB.
_mock_database = types.ModuleType("database")
_mock_database.models = types.SimpleNamespace(Simulation=object)  # minimal placeholder
sys.modules.setdefault("database", _mock_database)
# ------------------------------------------------------------------------------

# Now safe to import from the package
from backend.services.portfolio_simulator.portfolio_simulator import (
    PortfolioSimulator,
    PortfolioWeightCalculator,
    SimulationResult,
)


@pytest.fixture
def sample_data():
    """Generate synthetic price data for 3 assets over 200 business days."""
    np.random.seed(42)
    days = 200
    tickers = ["AAA", "BBB", "CCC"]

    data = pd.DataFrame(
        np.cumprod(
            1 + np.random.normal(0.0005, 0.01, (days, len(tickers))), axis=0
        )
        * 100,
        columns=tickers,
        index=pd.date_range("2020-01-01", periods=days, freq="B"),
    )
    return data


def test_weight_calculator_basic(sample_data):
    """Weights should be positive and sum to 1 for typical risk profiles."""
    calc = PortfolioWeightCalculator()

    for risk_score, label in [(20, "conservative"), (50, "moderate"), (80, "aggressive")]:
        weights = calc.calculate_weights(sample_data.columns.tolist(), risk_score, label)
        assert isinstance(weights, np.ndarray)
        assert np.isclose(np.sum(weights), 1.0)  # weights must sum to 1
        assert np.all(weights > 0)               # no negative weights


def test_portfolio_simulator_growth(sample_data):
    """Run a basic growth simulation and check core outputs exist."""
    sim = PortfolioSimulator()
    weights = np.array([0.4, 0.3, 0.3])

    result = sim.simulate_growth(
        data=sample_data,
        weights=weights,
        lump_sum=10_000,
        monthly_contribution=500,
        timeframe_years=1,
    )

    assert isinstance(result, SimulationResult)
    assert result.timeline_data, "Timeline data should not be empty"
    assert result.portfolio_metrics.ending_value > 0
    # Basic sanity: annualized_return is present and finite
    assert np.isfinite(result.portfolio_metrics.annualized_return)


def test_max_drawdown():
    """Check max drawdown calculation on a simple sequence."""
    sim = PortfolioSimulator()
    values = [100, 120, 80, 90, 110]  # peak=120, trough=80 -> drawdown ≈ 0.333
    dd = sim._calculate_max_drawdown(values)
    assert np.isclose(dd, 0.333, atol=0.01)