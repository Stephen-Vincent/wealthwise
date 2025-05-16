# utils/load_csv_data.py

import pandas as pd
import os
from backend.utils.fetch_yfinance import fetch_and_save_stock_data

def load_stock_data(ticker, data_dir="data/stocks"):
    """
    Loads historical stock data from a CSV file for a given ticker symbol.
    If the CSV file doesn't exist or is unreadable, fetches the data using yfinance.
    """
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if not df.empty:
                print(f"📄 Loaded {ticker} data from CSV.")
                return df
            else:
                print(f"⚠️ Empty CSV for {ticker}, refetching.")
        except Exception as e:
            print(f"⚠️ Error reading {ticker}.csv: {e}")
    
    # If file doesn't exist or is invalid, fetch from yfinance
    print(f"📡 Fetching data for {ticker} from yfinance...")
    return fetch_and_save_stock_data(ticker)