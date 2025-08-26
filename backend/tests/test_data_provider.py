# backend/tests/test_data_provider.py
"""
Unit tests for data_provider.py with config initialization.

We load the module directly from its file path but set an appropriate package
context so its relative imports (e.g. `from .config`) work. Then we ensure
the app config is initialized before instantiating MarketDataProvider or
DataQualityAnalyzer.

Covers:
- Synthetic data generation (no yfinance dependency)
- Supported ticker metadata and info lookups
- Data quality analysis on synthetic data
"""

import os
import sys
import types
import importlib.util
import pytest
import pandas as pd


# ---------- lightweight loader ----------
def _load_data_provider_module():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    module_path = os.path.join(
        project_root, "backend", "services", "portfolio_simulator", "data_provider.py"
    )
    pkg_name = "backend.services.portfolio_simulator"
    full_name = f"{pkg_name}.data_provider"

    if pkg_name not in sys.modules:
        pkg_mod = types.ModuleType(pkg_name)
        pkg_mod.__path__ = [
            os.path.join(project_root, "backend", "services", "portfolio_simulator")
        ]
        sys.modules[pkg_name] = pkg_mod

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    for name, path_parts in [
        ("backend", ["backend"]),
        ("backend.services", ["backend", "services"]),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(project_root, *path_parts)]
            sys.modules[name] = m

    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# Load once
_data_provider = _load_data_provider_module()
MarketDataProvider = _data_provider.MarketDataProvider
DataQualityAnalyzer = _data_provider.DataQualityAnalyzer

# Import config initializer
from backend.services.portfolio_simulator.config import initialize_config


@pytest.fixture(scope="session", autouse=True)
def setup_config():
    """Ensure config is initialized before any tests run."""
    initialize_config()
    yield


@pytest.fixture
def provider():
    return MarketDataProvider()


def test_synthetic_price_series(provider):
    tickers = ["VTI", "VOO"]
    df = provider._synthetic_price_series(tickers, years=1, meta={})

    assert isinstance(df, pd.DataFrame)
    assert all(t in df.columns for t in tickers)
    assert len(df) > 50
    assert (df > 0).all().all()


def test_supported_tickers(provider):
    tickers = provider.get_supported_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert all(isinstance(t, str) and t for t in tickers)


def test_get_ticker_info(provider):
    tickers = provider.get_supported_tickers()
    info = provider.get_ticker_info(tickers[0])
    assert info is None or isinstance(info, dict)


def test_data_quality_analyzer():
    analyzer = DataQualityAnalyzer()
    provider = MarketDataProvider()
    df = provider._synthetic_price_series(["VTI"], years=1, meta={})
    result = analyzer.analyze_data_quality(df)

    for key in ("overall_quality", "tickers", "completeness", "volatility"):
        assert key in result