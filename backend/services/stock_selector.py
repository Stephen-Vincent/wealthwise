# backend/services/stock_selector.py

RISK_PORTFOLIO = {
    "Cautious": ["AAPL", "META", "GOOGL"],
    "Balanced": ["MSFT", "KO", "F"],
    "Adventurous": ["TSLA", "SBUX", "HOG"]
}

def select_stocks(risk_level: str) -> list[str]:
    """
    Returns a list of stocks based on the user's risk profile.

    Parameters:
    - risk_level (str): The user's selected risk profile ("Cautious", "Balanced", or "Adventurous").

    Returns:
    - list[str]: A list of stock tickers corresponding to the given risk level.
    """
    print(f"[Stock Selector] Risk level received: {risk_level}")
    return RISK_PORTFOLIO.get(risk_level, RISK_PORTFOLIO["Balanced"])