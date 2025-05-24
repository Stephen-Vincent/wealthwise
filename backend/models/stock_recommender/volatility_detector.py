import yfinance as yf
import pandas as pd

def get_stock_history(ticker: str, period="10y", interval="1d") -> pd.DataFrame:
    """
    Fetches historical stock data for a given ticker.
    
    Parameters:
        ticker (str): The stock ticker symbol.
        period (str): The period over which to fetch data. Default is "10y".
        interval (str): The data interval. Default is "1d".
        
    Returns:
        pd.DataFrame: DataFrame containing historical stock data.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

# Example usage:
# history = get_stock_history("AAPL", period="10y")
