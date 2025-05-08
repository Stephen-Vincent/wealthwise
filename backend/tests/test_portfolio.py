import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from services.portfolio_simulator import simulate_portfolio

def test_simulate_portfolio_basic_lump_sum():
    user_input = {
        "stocks": ["AAPL", "MSFT"],
        "lump_sum": 10000,
        "monthly": 0,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    # Check the output keys exist
    assert "final_balance" in result
    assert "timeline" in result
    assert isinstance(result["timeline"], list)
    assert result["final_balance"] > 10000  # likely grown over time

    print("\nTest Passed ✅")
    print(f"Final Balance: ${result['final_balance']:.2f}")
    print(f"Timeline Sample: {result['timeline'][:3]}")

def test_simulate_portfolio_monthly_investment():
    user_input = {
        "risk": "Balanced",
        "lump_sum": 0,
        "monthly": 500,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    assert "final_balance" in result
    assert "timeline" in result
    assert isinstance(result["timeline"], list)
    assert result["final_balance"] > 0  # some growth expected

    print("\nMonthly Investment Test Passed ✅")
    print(f"Final Balance: ${result['final_balance']:.2f}")
    print(f"Timeline Sample: {result['timeline'][:3]}")

def test_simulate_portfolio_cautious_risk():
    user_input = {
        "risk": "Cautious",
        "lump_sum": 10000,
        "monthly": 0,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    assert "final_balance" in result
    assert "timeline" in result
    assert isinstance(result["timeline"], list)

    print("\nCautious Risk Test Passed ✅")
    print(f"Final Balance: ${result['final_balance']:.2f}")
    print(f"Tickers: {list(result['portfolio'].keys())}")

def test_simulate_portfolio_adventurous_risk():
    user_input = {
        "risk": "Adventurous",
        "lump_sum": 10000,
        "monthly": 0,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    assert "final_balance" in result
    assert "timeline" in result
    assert isinstance(result["timeline"], list)

    print("\nAdventurous Risk Test Passed ✅")
    print(f"Final Balance: ${result['final_balance']:.2f}")
    print(f"Tickers: {list(result['portfolio'].keys())}")


def test_simulate_portfolio_zero_investment():
    user_input = {
        "risk": "Balanced",
        "lump_sum": 0,
        "monthly": 0,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    assert result["final_balance"] == 0
    assert isinstance(result["timeline"], list)
    print("\nZero Investment Test Passed ✅")


def test_simulate_portfolio_invalid_risk_profile():
    user_input = {
        "risk": "NonExistentRisk",
        "lump_sum": 10000,
        "monthly": 0,
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
    }

    result = simulate_portfolio(user_input)

    assert result["final_balance"] > 0
    assert isinstance(result["timeline"], list)
    print("\nInvalid Risk Profile Fallback Test Passed ✅")


from datetime import datetime
def test_simulate_portfolio_future_start_date():
    future_date = (datetime.today().replace(year=datetime.today().year + 1)).strftime("%Y-%m-%d")
    user_input = {
        "risk": "Balanced",
        "lump_sum": 10000,
        "monthly": 0,
        "start_date": future_date,
        "end_date": future_date,
    }

    result = simulate_portfolio(user_input)

    assert result["final_balance"] == 10000
    assert result["timeline"] == []
    print("\nFuture Start Date Test Passed ✅")