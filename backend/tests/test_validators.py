# backend/tests/test_validators.py

import sys
from pathlib import Path
import types
import pytest
from unittest.mock import MagicMock, patch

# ---- Ensure project root is importable ----
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---- Insert fake "database" module to prevent ImportError ----
sys.modules["database"] = types.ModuleType("database")
sys.modules["database.models"] = types.ModuleType("database.models")

# ---- Import modules via package path (safe now) ----
from backend.services.portfolio_simulator import validators as _validators
from backend.services.portfolio_simulator.exceptions import (
    ValidationError,
    SecurityError,
)

InputValidator = _validators.InputValidator

# ---- Fixtures ----
@pytest.fixture
def mock_config():
    """Mock configuration object with only the fields validators need."""
    mock = MagicMock()
    mock.simulation.min_investment_amount = 1000
    mock.simulation.max_investment_amount = 1_000_000
    mock.simulation.min_timeframe_years = 1
    mock.simulation.max_timeframe_years = 50
    mock.security.max_filename_length = 50
    return mock

@pytest.fixture
def validator(mock_config):
    """Provide an InputValidator with get_config patched inside validators module."""
    with patch.object(_validators, "get_config", return_value=mock_config):
        yield InputValidator()

# ---- Tests ----
def test_validate_goal_success(validator):
    result = validator.validate_goal("Retirement Plan")
    assert isinstance(result, str)
    assert "Retirement" in result

def test_validate_goal_too_short(validator):
    with pytest.raises(ValidationError):
        validator.validate_goal("Hi")

def test_validate_target_value_within_bounds(validator):
    assert validator.validate_target_value(50_000) == 50_000.0

def test_validate_target_value_too_low(validator):
    with pytest.raises(ValidationError):
        validator.validate_target_value(100)

def test_validate_timeframe_valid(validator):
    assert validator.validate_timeframe(10) == 10

def test_validate_timeframe_out_of_bounds(validator):
    with pytest.raises(ValidationError):
        validator.validate_timeframe(0)

def test_validate_risk_score_valid(validator):
    assert validator.validate_risk_score(50) == 50

def test_validate_risk_score_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_risk_score(200)

def test_validate_risk_label_valid(validator):
    assert validator.validate_risk_label("moderate") == "Moderate"

def test_validate_risk_label_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_risk_label("SuperHighRisk")

def test_validate_investment_amount_valid(validator):
    assert validator.validate_investment_amount(5_000) == 5_000.0

def test_validate_investment_amount_negative(validator):
    with pytest.raises(ValidationError):
        validator.validate_investment_amount(-100)

def test_validate_experience_years_valid(validator):
    assert validator.validate_experience_years(10) == 10

def test_validate_experience_years_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_experience_years(-1)

def test_validate_income_bracket_valid(validator):
    assert validator.validate_income_bracket("high") == "high"

def test_validate_income_bracket_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_income_bracket("ultra_rich")

def test_validate_user_id_valid(validator):
    assert validator.validate_user_id(123) == 123

def test_validate_user_id_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_user_id(-5)

def test_validate_ticker_symbols_valid(validator):
    result = validator.validate_ticker_symbols(["AAPL", "msft"])
    assert set(result) == {"AAPL", "MSFT"}

def test_validate_ticker_symbols_invalid(validator):
    with pytest.raises(ValidationError):
        validator.validate_ticker_symbols(["INVALID!!!"])

def test_validate_weights_valid(validator):
    result = validator.validate_weights([0.5, 0.5])
    assert sum(result) == pytest.approx(1.0)

def test_validate_weights_negative(validator):
    with pytest.raises(ValidationError):
        validator.validate_weights([0.5, -0.1])

def test_sanitize_string_valid(validator):
    result = validator.sanitize_string("Hello World!")
    assert isinstance(result, str)

def test_sanitize_string_sql_injection(validator):
    with pytest.raises(SecurityError):
        validator.sanitize_string("DROP TABLE users;")

def test_sanitize_filename_valid(validator):
    result = validator.sanitize_filename("report.pdf")
    assert result.endswith(".pdf")

def test_sanitize_filename_path_traversal(validator):
    with pytest.raises(SecurityError):
        validator.sanitize_filename("../etc/passwd")