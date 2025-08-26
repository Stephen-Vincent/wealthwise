# backend/tests/test_ai_analysis.py
import sys
import types
import pytest

# --- Patch out the problematic imports before ai_analysis is loaded ---
# Fake database.models
fake_database = types.ModuleType("database")
fake_database.models = types.SimpleNamespace()
sys.modules["database"] = fake_database

# Fake news_analysis
fake_news = types.ModuleType("backend.services.portfolio_simulator.news_analysis")
fake_news.NewsAnalysisService = object  # placeholder
sys.modules["backend.services.portfolio_simulator.news_analysis"] = fake_news

# Now import your service
from backend.services.portfolio_simulator.ai_analysis import AIAnalysisService


@pytest.fixture
def ai_service():
    return AIAnalysisService()


def test_extract_symbols_from_holdings(ai_service):
    portfolio = {"holdings": [{"symbol": "AAPL"}, {"symbol": "TSLA"}]}
    symbols = ai_service.extract_symbols_from_portfolio(portfolio)
    assert set(symbols) == {"AAPL", "TSLA"}


def test_extract_symbols_from_positions(ai_service):
    portfolio = {"positions": [{"ticker": "msft"}, {"ticker": "goog"}]}
    symbols = ai_service.extract_symbols_from_portfolio(portfolio)
    assert set(symbols) == {"MSFT", "GOOG"}


def test_extract_symbols_from_stocks(ai_service):
    portfolio = {"stocks": ["amzn", "meta"]}
    symbols = ai_service.extract_symbols_from_portfolio(portfolio)
    assert set(symbols) == {"AMZN", "META"}


def test_extract_symbols_from_stocks_picked(ai_service):
    portfolio = {"stocks_picked": [{"symbol": "nvda"}, {"ticker": "intc"}]}
    symbols = ai_service.extract_symbols_from_portfolio(portfolio)
    assert set(symbols) == {"NVDA", "INTC"}


def test_extract_symbols_empty(ai_service):
    portfolio = {"unrelated": []}
    symbols = ai_service.extract_symbols_from_portfolio(portfolio)
    assert symbols == []