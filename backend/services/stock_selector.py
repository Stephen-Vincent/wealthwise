# backend/services/stock_selector.py

from .risk_assessment import assess_risk_score

RISK_PORTFOLIO = {
    "Cautious": ["AAPL", "META", "GOOGL"],
    "Balanced": ["MSFT", "KO", "F"],
    "Adventurous": ["TSLA", "SBUX", "HOG"]
}

def select_stocks(user_data: dict) -> list[str]:
    """
    Returns a list of stocks based on the user's data by first assessing their risk profile.

    Parameters:
    - user_data (dict): The user's onboarding data.

    Returns:
    - list[str]: A list of stock tickers corresponding to the computed risk level.
    """
    risk_level = assess_risk_score(user_data)
    print(f"[Stock Selector] Assessed risk level: {risk_level}")
    return RISK_PORTFOLIO.get(risk_level, RISK_PORTFOLIO["Balanced"])

def get_stocks_for_risk_profile(risk_profile: str) -> list[str]:
    """
    Returns a list of stock tickers for a given risk profile.

    Parameters:
    - risk_profile (str): One of "Cautious", "Balanced", or "Adventurous".

    Returns:
    - list[str]: A list of stock tickers.
    """
    return RISK_PORTFOLIO.get(risk_profile, RISK_PORTFOLIO["Balanced"])