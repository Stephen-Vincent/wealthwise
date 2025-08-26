"""
Test suite for AIRecommendationService and SHAPDataProcessor.

These tests focus on ensuring the service can:
- Initialize properly
- Return fallback recommendations when AI models are unavailable
- Generate explanations for stocks
- Process SHAP explanations cleanly
"""

import os
import sys
import types
import pytest

# --- Ensure project root is on sys.path (so "backend" imports resolve) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Mock missing 'database' dependency so package import doesn't crash ---
# portfolio_simulator/__init__.py imports main_service -> database_service -> database.models
if "database" not in sys.modules:
    mock_db = types.ModuleType("database")
    mock_db.models = types.SimpleNamespace()  # minimal stub to satisfy "from database import models"
    sys.modules["database"] = mock_db

# Now safe to import the service
from backend.services.portfolio_simulator.ai_recommendation_service import (
    AIRecommendationService,
    SHAPDataProcessor,
)


@pytest.fixture
def ai_service():
    """Fixture to create an AIRecommendationService instance."""
    return AIRecommendationService()


def test_fallback_recommendations(ai_service):
    """Service should provide fallback recommendations when WealthWise AI is unavailable."""
    result = ai_service._get_fallback_recommendations(
        target_value=50000,
        timeframe_years=10,
        risk_score=45,
        risk_label="Moderate",
    )
    assert "stocks" in result
    assert isinstance(result["stocks"], list)
    assert len(result["stocks"]) > 0
    assert result["method"] == "fallback_rule_based"


def test_stock_explanation(ai_service):
    """get_stock_explanation should return a string explanation."""
    recs = ai_service._get_fallback_recommendations(100000, 10, 60, "Moderate")
    explanation = ai_service.get_stock_explanation(recs["stocks"][0], recs)
    assert isinstance(explanation, str)
    assert len(explanation) > 5


def test_shap_data_processor_cleaning():
    """SHAPDataProcessor should clean raw SHAP explanation dictionaries."""
    processor = SHAPDataProcessor()

    raw_explanation = {
        "feature_contributions": {
            "risk_score": 0.8,
            "timeframe": -0.5,
            "monthly_contribution": [200],
        },
        "base_value": [45.0],
        "portfolio_quality_score": 85.0,
    }

    cleaned = processor.clean_shap_explanation(raw_explanation)

    assert "feature_contributions" in cleaned
    assert isinstance(cleaned["feature_contributions"], dict)
    assert "human_readable_explanation" in cleaned
    assert "overall" in cleaned["human_readable_explanation"]


def test_basic_goal_analysis_and_feasibility(ai_service):
    """Check that basic goal analysis and feasibility produce sensible values."""
    goal_analysis = ai_service._calculate_basic_goal_analysis(50000, 10, 50)
    feasibility = ai_service._assess_basic_feasibility(50000, 10, 50)

    assert "required_return_percent" in goal_analysis
    assert "feasibility_score" in feasibility
    assert 0 <= feasibility["feasibility_score"] <= 100