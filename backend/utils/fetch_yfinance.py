import os
import yfinance as yf
import pandas as pd

def fetch_and_save_stock_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="10y")  # adjust as needed
        if df.empty:
            print(f"⚠️ No data found from yfinance for {ticker}")
            return pd.DataFrame()
        
        df.to_csv(f"backend/data/stocks/{ticker}.csv")
        print(f"✅ Downloaded and saved {ticker} data.")
        return df
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return pd.DataFrame()