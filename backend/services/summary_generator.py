# services/summary_generator.py

import datetime

def analyze_stock(ticker: str, start_date: str, end_date: str) -> str:
    """Detailed analysis for an individual stock."""
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return "❌ Invalid date format. Use YYYY-MM-DD."

    return (
        f"🔍 Analysis for {ticker.upper()} ({start_date} to {end_date}):\n"
        "- The stock saw movement due to tech sector momentum.\n"
        "- Q2 earnings and new product launches influenced price.\n"
        "- Macroeconomic events (e.g., interest rates) had impact.\n"
        "→ This is a simulated summary. Real LLM analysis to follow.\n"
    )


def summarize_portfolio(simulation: dict) -> str:
    """Generate a human-readable summary of the entire portfolio simulation."""
    results = simulation.get("results", {})
    stocks = results.get("stocks_picked", [])

    if not simulation or not stocks:
        return "⚠️ No portfolio data available to summarize."

    name = simulation.get("name", "Unnamed Portfolio")
    goal = simulation.get("goal", "No goal specified")
    target_value = simulation.get("target_value", "N/A")
    lump_sum = simulation.get("lump_sum", 0)
    monthly = simulation.get("monthly", 0)
    timeframe = simulation.get("timeframe", "N/A")
    target_achieved = simulation.get("target_achieved", False)
    income_bracket = simulation.get("income_bracket", "N/A")
    risk_score = simulation.get("risk_score", "N/A")
    risk_label = simulation.get("risk_label", "N/A")
    start_value = results.get("starting_value", "N/A")
    end_value = results.get("end_value", "N/A")

    total_invested = lump_sum + (monthly * 12 * int(timeframe))
    try:
        portfolio_return = round(((end_value - total_invested) / total_invested) * 100, 2)
    except (TypeError, ZeroDivisionError):
        portfolio_return = "N/A"

    stock_list = ", ".join(
        stock.get("symbol", "UNKNOWN") for stock in stocks
    )

    status = "achieved" if target_achieved else "not achieved"

    # Format monetary values with pound symbol and two decimals
    lump_sum_str = f"£{lump_sum:,.2f}"
    monthly_str = f"£{monthly:,.2f}"
    target_value_str = f"£{target_value:,.2f}" if isinstance(target_value, (int, float)) else target_value

    if isinstance(start_value, (int, float)):
        start_value_str = f"£{start_value:,.2f}"
    else:
        start_value_str = start_value

    if isinstance(end_value, (int, float)):
        end_value_str = f"£{end_value:,.2f}"
    else:
        end_value_str = end_value

    total_invested_str = f"£{total_invested:,.2f}"

    if isinstance(portfolio_return, (int, float)):
        portfolio_return_str = f"{portfolio_return:.2f}%"
    else:
        portfolio_return_str = portfolio_return

    return (
        f"📊 Portfolio Summary:\n\n"
        f"The portfolio '{name}' was created with the goal of '{goal}' over a {timeframe}-year timeframe. "
        f"The user started with a lump sum of {lump_sum_str} and monthly contributions of {monthly_str}, aiming for a target value of {target_value_str}. "
        f"Based on an income bracket of '{income_bracket}', a risk score of {risk_score} categorized the portfolio as '{risk_label}' risk.\n\n"
        f"The selected stocks were: {stock_list}. The portfolio started at {start_value_str} and grew to {end_value_str}, "
        f"resulting in an overall return of {portfolio_return_str} based on a total investment of {total_invested_str}. The target was {target_value_str}.\n\n"
        f"→ This is a mock summary. Future versions will include real insights."
    )
