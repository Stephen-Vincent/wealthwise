import yfinance as yf
import pandas as pd
from typing import List, Dict

def get_stock_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical stock data for the given list of symbols between start_date and end_date.
    
    Args:
        symbols (List[str]): List of stock symbols (e.g., ["AAPL", "GOOGL"])
        start_date (str): Start date in "YYYY-MM-DD" format
        end_date (str): End date in "YYYY-MM-DD" format

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping symbol to historical data DataFrame
    """
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        hist.index = pd.to_datetime(hist.index)  # Ensure datetime index
        hist = hist.resample('M').last()         # Keep only end-of-month data
        data[symbol] = pd.DataFrame(hist)
    return data

def get_current_price(symbol: str) -> float:
    """
    Fetch the current stock price for a single symbol.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL")

    Returns:
        float: Current stock price
    """
    ticker = yf.Ticker(symbol)
    price = ticker.info.get('currentPrice')
    return price