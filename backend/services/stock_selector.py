# backend/services/stock_selector.py

RISK_PORTFOLIO = {
    "Cautious": ["AAPL", "META", "GOOGL"],
    "Balanced": ["MSFT", "KO", "F"],
    "Adventurous": ["TSLA", "SBUX", "HOG"]
}

def select_stocks(risk_level):
    """Returns a list of stocks based on the risk level."""
    return RISK_PORTFOLIO.get(risk_level, RISK_PORTFOLIO["Balanced"])