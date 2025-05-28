import yfinance as yf
import pandas as pd

def get_stock_history(ticker: str, period="10y", interval="1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

def calculate_volatility(ticker: str, period="10y", interval="1d", window=30) -> float:
    """
    Calculates the latest rolling volatility (std of daily returns) for a stock.
    Returns a single float value representing the latest volatility.
    """
    df = get_stock_history(ticker, period, interval)

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        return 0.0

    df["returns"] = df["Close"].pct_change()
    df["rolling_volatility"] = df["returns"].rolling(window=window).std()

    latest_vol = df["rolling_volatility"].iloc[-1]
    print(f"🔍 Latest 30-day rolling volatility for {ticker}: {latest_vol:.4f}")

    return latest_vol