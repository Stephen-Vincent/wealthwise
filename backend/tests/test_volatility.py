import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models.stock_recommender.volatility_detector import calculate_volatility

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    for t in tickers:
        calculate_volatility(t)