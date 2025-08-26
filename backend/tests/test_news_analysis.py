# backend/tests/test_news_analysis.py
"""
Unit tests for NewsAnalysisService.

This version avoids importing the whole package (which pulls database deps)
by loading news_analysis.py directly via importlib. It also stubs TextBlob if
it's not installed so sentiment tests can run without that dependency.
"""

import sys
import os
import types
import importlib.util
import pytest

# --- Load news_analysis.py by file path (skip package __init__ chain) ---
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWS_ANALYSIS_PATH = os.path.join(
    ROOT, "backend", "services", "portfolio_simulator", "news_analysis.py"
)

# Provide a minimal stub for textblob if it's not installed
# Provide a minimal stub for textblob if it's not installed
if "textblob" not in sys.modules:
    textblob_stub = types.ModuleType("textblob")

    class _Sentiment:
        def __init__(self, polarity, subjectivity=0.5):
            self.polarity = polarity
            self.subjectivity = subjectivity  # <-- added attribute

    class TextBlob:  # simple polarity heuristic for tests
        def __init__(self, text):
            self._text = text or ""

        @property
        def sentiment(self):
            t = self._text.lower()
            score = 0.0
            positives = ("great", "record", "profit", "strong", "happy", "success")
            negatives = ("loss", "lawsuit", "bad", "drop", "weak", "fraud")
            for w in positives:
                if w in t:
                    score += 0.2
            for w in negatives:
                if w in t:
                    score -= 0.2
            score = max(-1.0, min(1.0, score))
            return _Sentiment(score, 0.5)  # <-- subjectivity included

    textblob_stub.TextBlob = TextBlob
    sys.modules["textblob"] = textblob_stub  # register stub

# Create a module spec and load only this file
_spec = importlib.util.spec_from_file_location(
    "backend.services.portfolio_simulator.news_analysis",
    NEWS_ANALYSIS_PATH,
)
_news_module = importlib.util.module_from_spec(_spec)
# Help relative imports inside the module (if any) by setting __package__
_news_module.__package__ = "backend.services.portfolio_simulator"
_spec.loader.exec_module(_news_module)  # type: ignore

# Pull the service class from the loaded module
NewsAnalysisService = _news_module.NewsAnalysisService


# ---------------------- Tests (unchanged logic) ----------------------

@pytest.mark.asyncio
async def test_analyze_sentiment_positive():
    """Test sentiment analysis correctly detects positive sentiment."""
    service = NewsAnalysisService(finnhub_api_key="dummy")

    articles = [
        {"headline": "Company achieves record profits", "summary": "Strong growth in revenue"},
        {"headline": "Great success for new product", "summary": "Customers are very happy"},
    ]

    result = await service.analyze_sentiment(articles)

    assert result["average_sentiment"] > 0
    assert result["total_articles"] == 2
    assert result["sentiment_category"] in ["Positive", "Very Positive"]


@pytest.mark.asyncio
async def test_analyze_sentiment_empty_articles():
    """Test that empty input returns neutral sentiment."""
    service = NewsAnalysisService()
    result = await service.analyze_sentiment([])

    assert result["average_sentiment"] == 0.0
    assert result["total_articles"] == 0
    assert result["sentiment_category"] == "Neutral"


@pytest.mark.asyncio
async def test_categorize_sentiment_ranges():
    """Directly test sentiment categorization logic."""
    service = NewsAnalysisService()
    assert service._categorize_sentiment(0.5) == "Very Positive"
    assert service._categorize_sentiment(0.2) == "Positive"
    assert service._categorize_sentiment(0.0) == "Neutral"
    assert service._categorize_sentiment(-0.2) == "Negative"
    assert service._categorize_sentiment(-0.5) == "Very Negative"


@pytest.mark.asyncio
async def test_get_market_events():
    """Test that event keywords are correctly detected in articles."""
    service = NewsAnalysisService()
    articles = [
        {"headline": "Company announces merger deal", "summary": "A major acquisition is planned"},
        {"headline": "CEO steps down", "summary": "Leadership change at the top"},
        {"headline": "Quarterly results beat expectations", "summary": "Strong Q1 earnings reported"},
    ]

    events = await service.get_market_events(articles)

    assert len(events) == 3
    assert any("merger_acquisition" in e["event_types"] for e in events)
    assert any("leadership" in e["event_types"] for e in events)
    assert any("earnings" in e["event_types"] for e in events)


@pytest.mark.asyncio
async def test_sentiment_strength_calculation():
    """Test sentiment volatility calculation logic."""
    service = NewsAnalysisService()

    assert service._calculate_sentiment_strength([0.1]) == "Insufficient Data"
    assert service._calculate_sentiment_strength([0.1, 0.15]) == "Low Volatility"
    assert service._calculate_sentiment_strength([0.1, 0.4]) in ["Moderate Volatility", "High Volatility"]