# utils/fetch_yfinance.py
import yfinance as yf

def fetch_historical_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]
    return data