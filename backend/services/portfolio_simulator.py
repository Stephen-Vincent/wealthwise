from datetime import datetime
from utils.fetch_yfinance import fetch_historical_data

RISK_PORTFOLIO = {
    "Cautious": ["KO", "JNJ", "PG"],
    "Balanced": ["MSFT", "V", "UNH"],
    "Adventurous": ["TSLA", "AMZN", "NVDA"],
}

def simulate_portfolio(user_input):
    risk = user_input.get("risk", "Balanced")
    lump_sum = float(user_input.get("lump_sum", 0))
    monthly = float(user_input.get("monthly", 0))
    start_date = user_input.get("start_date", "2019-01-01")
    end_date = user_input.get("end_date", datetime.today().strftime("%Y-%m-%d"))

    tickers = RISK_PORTFOLIO.get(risk, RISK_PORTFOLIO["Balanced"])
    data = fetch_historical_data(tickers, start_date, end_date)

    results = {}
    timeline = []

    if data.empty:
        return {
            "portfolio": {},
            "total_start": lump_sum,
            "total_end": lump_sum,
            "final_balance": lump_sum,
            "timeline": [],
        }

    total_months = max((datetime.strptime(end_date, "%Y-%m-%d").year - datetime.strptime(start_date, "%Y-%m-%d").year) * 12 + 
                       (datetime.strptime(end_date, "%Y-%m-%d").month - datetime.strptime(start_date, "%Y-%m-%d").month), 0)
    total_investment = lump_sum + (monthly * total_months)
    equal_allocation = total_investment / len(tickers)

    # Build timeline of daily total portfolio value
    for date in data.index:
        total_value = 0
        for ticker in tickers:
            prices = data[ticker].dropna()
            if date in prices.index and not prices.empty:
                start_price = prices.iloc[0]
                current_price = prices.at[date]
                months_since_start = (date.year - datetime.strptime(start_date, "%Y-%m-%d").year) * 12 + (date.month - datetime.strptime(start_date, "%Y-%m-%d").month)
                months_invested = max(0, min(months_since_start + 1, total_months))
                invested = lump_sum + (monthly * months_invested)
                shares = invested / len(tickers) / start_price
                total_value += shares * current_price
        timeline.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(total_value, 2)
        })

    # Final summary
    for ticker in tickers:
        prices = data[ticker].dropna()
        if len(prices) > 0:
            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
            growth = (end_price - start_price) / start_price
            final_value = total_investment * (1 + growth)
            results[ticker] = {
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "growth_pct": round(growth * 100, 2),
                "final_value": round(final_value, 2),
            }

    total_end = round(sum(r["final_value"] for r in results.values()), 2)

    return {
        "portfolio": results,
        "total_start": total_investment,
        "total_end": total_end,
        "final_balance": total_end,
        "timeline": timeline,
    }