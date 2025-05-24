from backend.services.stock_selector import get_stocks_for_risk_profile
from backend.services.risk_assessment import assess_risk_score
from datetime import datetime
import os
from backend.utils.load_csv_data import load_stock_data
from backend.utils.fetch_yfinance import fetch_and_save_stock_data
import pandas as pd
import traceback
import math


def simulate_portfolio(user_input):
    from dateutil.relativedelta import relativedelta
    try:
        # Parsing user inputs and setting default values
        risk, risk_score = assess_risk_score(user_input)
        print(f"🧠 Computed risk profile: {risk} ({risk_score})")
        lump_sum = float(user_input.get("lump_sum", 0))
      
        monthly = float(user_input.get("monthly", 0))
        actual_total_contributed = 0.0  # Track how much money is actually invested over time
        # Normalise and determine investment timeframe based on user selection
        timeframe_raw = user_input.get("timeframe", "1–5 years")
        # Normalize the timeframe string to a consistent format
        timeframe = timeframe_raw.strip().replace("–", "-")  # Normalize en dash to hyphen
        # Ensure the timeframe is in a consistent format
        target_value = float(user_input.get("target_value", 0))

        # <1 year = exactly 1 year, 1-5 years = past 5 years, 5+ years = past 10 years
        if timeframe.replace(" ", "") == "<1year":
            today = datetime.today()
            # Round down to the 1st of the current month
            today = today.replace(day=1)
            start_date = (today - relativedelta(years=1)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
        elif timeframe.strip() == "1-5 years":
            today = datetime.today()
            # Round down to the 1st of the current month
            today = today.replace(day=1)
            start_date = (today - relativedelta(years=5)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
        elif timeframe.strip() == "5+ years":
            today = datetime.today()
            # Round down to the 1st of the current month
            today = today.replace(day=1)
            start_date = (today - relativedelta(years=10)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
        else:
            today = datetime.today()
            # Round down to the 1st of the current month
            today = today.replace(day=1)
            start_date = (today - relativedelta(years=5)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

        print(f"⏱️ Simulating portfolio from {start_date} to {end_date} ({timeframe.strip()})")

        # Select a list of stock tickers based on the user's risk profile
        tickers = get_stocks_for_risk_profile(risk)

        # Load historical price data for each selected stock ticker
        stock_frames = []
        for ticker in tickers:
            try:
                df = load_stock_data(ticker)
                if df.empty:
                    df = fetch_and_save_stock_data(ticker)
                if not df.empty:
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    # ✅ Flatten MultiIndex columns if necessary
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    # Clean data: drop rows with NaNs or infinities in "Close"
                    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
                    df.dropna(subset=["Close"], inplace=True)
                    df = df.rename(columns={"Close": ticker})
                    stock_frames.append(df[[ticker]])
                else:
                    results[ticker] = {
                        "start_price": 0.0,
                        "end_price": 0.0,
                        "growth_pct": 0.0,
                        "final_value": 0.0,
                    }
            except Exception as e:
                print(f"❌ Failed to load or fetch data for {ticker}: {e}")
        if stock_frames:
            data = pd.concat(stock_frames, axis=1, join='inner')
            # Clean data: drop rows with NaNs or infinities
            data.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
            data.dropna(inplace=True)
        else:
            data = pd.DataFrame()

        results = {}
        timeline = []

        # Return empty data if price data is missing
        if data.empty:
            print("❌ Portfolio simulation aborted: No data loaded for selected tickers.")
            print("⚠️ Check if the tickers have data within the selected timeframe.")
            for ticker in tickers:
                results[ticker] = {
                    "start_price": 0.0,
                    "end_price": 0.0,
                    "growth_pct": 0.0,
                    "final_value": 0.0,
                }
            return {
                "portfolio": results,
                "total_start": lump_sum,
                "total_end": lump_sum,
                "final_balance": lump_sum,
                "timeline": [],
            }

        # Apply lump sum investment on the first valid trading day
        shares_held_by_ticker = {ticker: 0 for ticker in tickers}
        lump_sum_applied = False

        # --- Lump sum investment logic (for compatibility with new requirements) ---
        # This block is for reference to the new approach requested
        # (see message for copy-paste location in main.py)
        # ------------------------------------------------------
        # # Apply lump sum investment on the first day of the simulation
        # # This ensures shares are purchased up front even if the lump sum is 0
        # first_day = stock_prices.index[0]  # Get the earliest date in the price data
        # for stock in portfolio:
        #     price = stock_prices[stock].loc[first_day]['Close']
        #     invested_amount = allocation[stock] * lump_sum
        #     shares[stock] = invested_amount / price
        #     print(f"[DEBUG] Lump sum investment for {stock}: £{invested_amount:.2f} at £{price:.2f} per share. Shares bought: {shares[stock]:.4f}")
        # ------------------------------------------------------

        # Calculating total number of months in the investment period
        total_months = max((datetime.strptime(end_date, "%Y-%m-%d").year - datetime.strptime(start_date, "%Y-%m-%d").year) * 12 + 
                           (datetime.strptime(end_date, "%Y-%m-%d").month - datetime.strptime(start_date, "%Y-%m-%d").month), 0)

        equal_allocation = lump_sum / len(tickers)

        # Store parsed start and end dates for efficiency
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        target_reached_flag = False

        for i, date in enumerate(data.index):
            # Apply lump sum investment on the first valid trading day, even if the lump sum is zero.
            if not lump_sum_applied:
                lump_sum_per_ticker = lump_sum / len(tickers)
                for ticker in tickers:
                    price_today = data[ticker].loc[date]
                    shares_held_by_ticker[ticker] += lump_sum_per_ticker / price_today
                actual_total_contributed += lump_sum
              
                lump_sum_applied = True

           

            # Apply monthly contribution (fractional shares) on the first trading day of each month
            if monthly > 0 and (i == 0 or data.index[i - 1].month != date.month):
                monthly_per_ticker = monthly / len(tickers)
                for ticker in tickers:
                    price_at_contribution = data[ticker].loc[date]
                    shares_held_by_ticker[ticker] += monthly_per_ticker / price_at_contribution
                actual_total_contributed += monthly
              

            # Calculate portfolio value
            total_value = sum(shares_held_by_ticker[t] * data[t].loc[date] for t in tickers if date in data[t])
          
            # Determine if this date represents a contribution
            is_contribution_day = monthly > 0 and (i == 0 or data.index[i - 1].month != date.month)
            is_target_reached = False
            if not target_reached_flag and target_value > 0 and total_value >= target_value:
                is_target_reached = True
                target_reached_flag = True
                print(f"🎯 Target of £{target_value} reached on {date.strftime('%Y-%m-%d')} with portfolio value £{total_value:.2f}")
                print(f"🎯 Target of £{target_value} reached on {date.strftime('%Y-%m-%d')} with portfolio value £{total_value:.2f}")

            timeline.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(total_value, 2),
                "is_contribution": is_contribution_day,
                "target_reached": is_target_reached
            })

        # Calculate final values, growth, and construct a timeline of portfolio value over time
        final_total_value = timeline[-1]["value"] if timeline else 0.0
        ticker_values = {t: shares_held_by_ticker[t] * data[t].iloc[-1] for t in tickers}
        total_value_check = sum(ticker_values.values()) or 1  # avoid division by zero

        for ticker in tickers:
            prices = data[ticker].dropna()
            if len(prices) > 0:
                start_price = prices.iloc[0]
                end_price = prices.iloc[-1]
                growth = (end_price - start_price) / start_price
                proportional_value = final_total_value * (ticker_values[ticker] / total_value_check)
                results[ticker] = {
                    "start_price": round(start_price, 2),
                    "end_price": round(end_price, 2),
                    "growth_pct": round(growth * 100, 2),
                    "final_value": round(proportional_value, 2),
                }

        # Sanitize timeline values to avoid NaN or infinite values
        for point in timeline:
            val = point["value"]
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                point["value"] = 0.0

        # Use the final total value from the timeline as the true end value
        total_end = final_total_value

        # Checking if the target value is achieved
        target_achieved = total_end >= target_value if target_value > 0 else None
        return {
            "portfolio": results,
            "total_invested": round(actual_total_contributed, 2),  # Full amount invested including monthly contributions
            "total_end": round(total_end, 2),
            "final_balance": round(total_end, 2),
            "timeline": timeline,
            "target_achieved": target_achieved,
            "initial_investment": round(lump_sum, 2),
            "monthly_contribution": round(monthly, 2),
            "total_loss": round(max(0, actual_total_contributed - total_end), 2),
            "risk_profile": risk,
            "risk_score": risk_score,
        }
    except Exception as e:
        print("❌ Portfolio simulation encountered an error:", e)
        traceback.print_exc()
        return {
            "detail": "Portfolio simulation failed",
            "error": str(e)
        }