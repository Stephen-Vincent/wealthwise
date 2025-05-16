from datetime import datetime
import os
from backend.utils.load_csv_data import load_stock_data
from backend.utils.fetch_yfinance import fetch_and_save_stock_data
import pandas as pd
import traceback
import math

RISK_PORTFOLIO = {
    "Cautious": ["KO", "JNJ", "PG"],
    "Balanced": ["MSFT", "V", "UNH"],
    "Adventurous": ["TSLA", "AMZN", "NVDA"],
}

def simulate_portfolio(user_input):
    from dateutil.relativedelta import relativedelta
    try:
        # Extract user input values and set defaults
        risk = user_input.get("risk", "Balanced")
        lump_sum = float(user_input.get("lump_sum", 0))
        monthly = float(user_input.get("monthly", 0))
        timeframe = user_input.get("timeframe", "1–5 years")

        # Determine the start date based on the timeframe selected
        if timeframe == "<1 year":
            start_date = (datetime.today() - relativedelta(years=1)).strftime("%Y-%m-%d")
        elif timeframe == "1–5 years":
            start_date = (datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")
        elif timeframe == "5+ years":
            start_date = (datetime.today() - relativedelta(years=20)).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.today() - relativedelta(years=5)).strftime("%Y-%m-%d")

        # Use today's date as the default end date
        end_date = user_input.get("end_date", datetime.today().strftime("%Y-%m-%d"))

        # Get the list of tickers for the selected risk profile
        tickers = RISK_PORTFOLIO.get(risk, RISK_PORTFOLIO["Balanced"])

        # Load data from CSVs and align on dates
        stock_frames = []
        for ticker in tickers:
            try:
                df = load_stock_data(ticker)
                if df.empty:
                    print(f"📉 No local data for {ticker}, fetching from yfinance...")
                    df = fetch_and_save_stock_data(ticker)
                if not df.empty:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    df = df.rename(columns={"Close": ticker})
                    stock_frames.append(df[[ticker]])
                else:
                    print(f"⚠️ Still no data found for {ticker} after fetch.")
            except Exception as e:
                print(f"❌ Failed to load or fetch data for {ticker}: {e}")
        if stock_frames:
            data = pd.concat(stock_frames, axis=1, join='inner')
        else:
            data = pd.DataFrame()

        results = {}
        timeline = []

        # Return empty data if price data is missing
        if data.empty:
            print("❌ Portfolio simulation aborted: No data loaded for selected tickers.")
            return {
                "portfolio": {},
                "total_start": lump_sum,
                "total_end": lump_sum,
                "final_balance": lump_sum,
                "timeline": [],
            }

        # Calculate total number of months in the investment period
        total_months = max((datetime.strptime(end_date, "%Y-%m-%d").year - datetime.strptime(start_date, "%Y-%m-%d").year) * 12 + 
                           (datetime.strptime(end_date, "%Y-%m-%d").month - datetime.strptime(start_date, "%Y-%m-%d").month), 0)

        equal_allocation = lump_sum / len(tickers)

        # Generate a timeline of total portfolio value over time
        for i, date in enumerate(data.index):
            total_value = 0
            months_since_start = (date.year - datetime.strptime(start_date, "%Y-%m-%d").year) * 12 + (date.month - datetime.strptime(start_date, "%Y-%m-%d").month)
            months_invested = max(0, min(months_since_start + 1, total_months))
            invested = lump_sum + (monthly * months_invested)

            for ticker in tickers:
                prices = data[ticker].dropna()
                if date in prices.index and not prices.empty:
                    start_price = prices.iloc[0]
                    current_price = prices.loc[date]
                    shares = invested / len(tickers) / start_price
                    total_value += shares * current_price
            timeline.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(total_value, 2)
            })

        # Generate final portfolio summary for each stock
        for ticker in tickers:
            prices = data[ticker].dropna()
            if len(prices) > 0:
                start_price = prices.iloc[0]
                end_price = prices.iloc[-1]
                growth = (end_price - start_price) / start_price
                final_value = (lump_sum + (monthly * total_months)) * (1 + growth)
                results[ticker] = {
                    "start_price": round(start_price, 2),
                    "end_price": round(end_price, 2),
                    "growth_pct": round(growth * 100, 2),
                    "final_value": round(final_value, 2),
                }

        # Sanitize timeline values
        for point in timeline:
            val = point["value"]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                point["value"] = 0.0

        # Sum the final value of all stocks for total ending balance
        total_end = round(sum(r["final_value"] for r in results.values()), 2)

        return {
            "portfolio": results,
            "total_start": lump_sum + (monthly * total_months),
            "total_end": total_end,
            "final_balance": total_end,
            "timeline": timeline,
        }
    except Exception as e:
        print("❌ Portfolio simulation encountered an error:", e)
        traceback.print_exc()
        return {
            "detail": "Portfolio simulation failed",
            "error": str(e)
        }