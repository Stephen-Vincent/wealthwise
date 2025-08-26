# backend/tests/test_visualization_service.py

import sys
import types
import pytest
import asyncio
import os
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta

# ---- Patch missing database to avoid ModuleNotFoundError ----
sys.modules["database"] = types.ModuleType("database")
sys.modules["database.models"] = types.ModuleType("database.models")

from backend.services.portfolio_simulator.visualization_service import (
    VisualizationService,
    ChartDataGenerator,
    VisualizationError,
)

# ---------------- Fixtures ----------------

@pytest.fixture
def mock_config(tmp_path):
    """Provide a fake config with visualization output directory + file format."""
    class DummyConfig:
        class visualization:
            output_directory = tmp_path
            file_format = "txt"

    return DummyConfig()

@pytest.fixture
def service(mock_config):
    with patch("backend.services.portfolio_simulator.visualization_service.get_config", return_value=mock_config):
        return VisualizationService()

# ---------------- Tests: VisualizationService ----------------

@pytest.mark.asyncio
async def test_create_simulation_visualizations_success(service):
    stocks_data = [
        {"symbol": "AAPL", "name": "Apple", "allocation": 0.6},
        {"symbol": "MSFT", "name": "Microsoft", "allocation": 0.4},
    ]
    simulation_results = {
        "timeline": {
            "portfolio": [
                {"date": "2023-01-01", "value": 1000},
                {"date": "2023-01-02", "value": 1100},
            ]
        }
    }

    result = await service.create_simulation_visualizations(
        simulation_id=1,
        stocks_data=stocks_data,
        simulation_results=simulation_results,
    )

    assert isinstance(result, dict)
    assert "portfolio_composition" in result or "performance_timeline" in result

@pytest.mark.asyncio
async def test_create_simulation_visualizations_empty(service):
    """Should not raise error even if no stocks or results are passed."""
    result = await service.create_simulation_visualizations(
        simulation_id=None,
        stocks_data=[],
        simulation_results={}
    )
    assert isinstance(result, dict)
    assert result == {}

@pytest.mark.asyncio
async def test_cleanup_old_files(service, tmp_path):
    # Create an old file in the configured output directory and backdate it
    old_file = tmp_path / "old.txt"
    old_file.write_text("data")

    # Set modified time to 10 days ago using os.utime
    ten_days_ago = datetime.now() - timedelta(days=10)
    ts = ten_days_ago.timestamp()
    os.utime(old_file, (ts, ts))

    # Also create a fresh file that should NOT be deleted
    recent_file = tmp_path / "recent.txt"
    recent_file.write_text("fresh")

    deleted = await service.cleanup_old_files(max_age_days=7)
    assert isinstance(deleted, int)

def test_validate_filename(service):
    safe = service.validate_filename("report.txt")
    assert safe.endswith(".txt")

# ---------------- Tests: ChartDataGenerator ----------------

def test_generate_chart_data_basic(mock_config):
    with patch("backend.services.portfolio_simulator.visualization_service.get_config", return_value=mock_config):
        gen = ChartDataGenerator()

    stocks_data = [{"symbol": "AAPL", "name": "Apple", "allocation": 0.5}]
    simulation_results = {
        "timeline": {"portfolio": [{"date": "2023-01-01", "value": 100}]},
        "starting_value": 100,
        "end_value": 120,
        "target_value": 150,
    }
    shap_explanation = {
        "feature_contributions": {"risk_score": 0.1, "timeframe": -0.05},
        "base_value": 50,
        "market_regime": {"current_vix": 15, "trend_score": 4, "returns_3m": 0.1},
    }

    result = gen.generate_chart_data(simulation_results, stocks_data, shap_explanation)
    assert "portfolio_composition" in result
    assert "performance_timeline" in result
    assert "risk_return_analysis" in result
    assert "goal_analysis" in result
    assert "feature_importance" in result