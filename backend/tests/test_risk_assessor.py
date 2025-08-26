# backend/tests/test_risk_assessor.py

import pytest
import types
import sys

# --- Test shim: mock the optional `database` package to avoid real DB imports ---
if "database" not in sys.modules:
    sys.modules["database"] = types.ModuleType("database")
if "database.models" not in sys.modules:
    sys.modules["database.models"] = types.ModuleType("database.models")

from backend.services.portfolio_simulator import risk_assessor


class DummyModel:
    """Mock model to simulate predictions."""
    def predict(self, X):
        # Always return a fixed score for predict
        return [50]


@pytest.fixture(autouse=True)
def mock_pickle(monkeypatch):
    """Automatically patch pickle.load to return DummyModel."""
    def fake_load(f):
        return DummyModel()

    monkeypatch.setattr(risk_assessor.pickle, "load", fake_load)
    yield


def test_calculate_user_risk_valid_input():
    """Ensure risk assessment runs end-to-end with valid input."""
    sim_input = {
        "years_of_experience": 5,
        "loss_tolerance": "wait_and_see",
        "panic_behavior": "no_never",
        "financial_behavior": "invest_all",
        "engagement_level": "monthly",
        "goal": "retirement",
        "target_value": 100000,
        "lump_sum": 10000,
        "monthly": 500,
        "timeframe": 10,
        "income_bracket": "medium",
    }

    result = risk_assessor.calculate_user_risk(sim_input)

    assert "risk_score" in result
    assert "risk_level" in result
    assert "recommended_stock_allocation" in result
    assert "recommended_bond_allocation" in result
    assert isinstance(result["risk_score"], float)


def test_missing_required_fields():
    """Should raise error if required fields are missing."""
    bad_input = {"goal": "retirement"}
    with pytest.raises(risk_assessor.RiskAssessmentError):
        risk_assessor.calculate_user_risk(bad_input)


def test_generate_risk_profile_levels():
    """Risk profile generation should match thresholds."""
    for score, expected_level in [
        (10, "Ultra Conservative"),
        (20, "Conservative"),
        (35, "Moderate Conservative"),
        (50, "Moderate"),
        (65, "Moderate Aggressive"),
        (80, "Aggressive"),
        (95, "Ultra Aggressive"),
    ]:
        profile = risk_assessor.generate_risk_profile(score, {"years_of_experience": 1})
        assert profile["risk_level"] == expected_level


def test_generate_risk_explanation():
    """Explanation should include context from inputs."""
    explanation = risk_assessor.generate_risk_explanation(
        50,
        {"years_of_experience": 0, "investment_goal": "retirement", "loss_tolerance": "buy_more", "timeframe": 20},
        "Moderate",
    )
    assert "new investor" in explanation.lower() or "retirement" in explanation.lower()


def test_legacy_wrapper():
    """Legacy wrapper should map to simplified labels."""
    sim_input = {
        "years_of_experience": 10,
        "goal": "retirement",
        "target_value": 50000,
        "timeframe": 5,
        "income_bracket": "medium",
    }
    score, label = risk_assessor.calculate_user_risk_legacy(sim_input)
    assert isinstance(score, int)
    assert label in ["Low", "Medium", "High"]