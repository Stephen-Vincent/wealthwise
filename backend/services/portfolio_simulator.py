from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.orm import Session
from database import models
import yfinance as yf
import numpy as np
import pandas as pd

# Import your AI summary generation function
from .summary_generator import generate_ai_summary

def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Simulate a user's portfolio using the provided onboarding input.

    This function receives onboarding data from the API layer, which already includes:
    - User's investment preferences and financial information.
    - A pre-calculated `risk_score` and `risk_label` returned from the risk model.

    It uses these inputs to:
    - Get AI recommended stocks.
    - Simulate portfolio growth over time.
    - Determine whether the user's target goal is reached.
    - Generate an AI educational summary explaining stock picks and market volatility.
    - Save simulation results and return data for frontend consumption.
    """
    
    user_data = {
        "experience": sim_input.get("experience"),
        "goal": sim_input.get("goal"),
        "target_value": sim_input.get("target_value"),
        "lump_sum": sim_input.get("lump_sum"),
        "monthly": sim_input.get("monthly"),
        "timeframe": sim_input.get("timeframe"),
        "income_bracket": sim_input.get("income_bracket")
    }

    risk_score = sim_input.get("risk_score")
    risk_label = sim_input.get("risk_label")

    # Get AI recommended stocks from your stock model
    from ai_models.train_stock_model import train_and_recommend
    tickers = train_and_recommend(
        target_value=user_data["target_value"],
        timeframe=user_data["timeframe"],
        risk_score=risk_score
    )
    if isinstance(tickers, str):
        tickers = tickers.split(',')

    allocation = round(1 / len(tickers), 2)
    stocks_picked = [{"symbol": ticker, "name": ticker, "allocation": allocation} for ticker in tickers]

    lump_sum = float(user_data.get("lump_sum") or 0)
    monthly = float(user_data.get("monthly") or 0)
    timeframe = int(user_data.get("timeframe") or 0)

    # Dates for data download
    start_date = (datetime.today() - timedelta(days=timeframe * 365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Download stock prices and simulate portfolio growth
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    weights = np.ones(len(tickers)) / len(tickers)
    normalized = data / data.iloc[0]
    weighted = normalized.dot(weights)

    portfolio_values = []
    contributions = []
    current_value = lump_sum
    total_contributions = lump_sum

    for i, (date, growth_factor) in enumerate(weighted.items()):
        if i % 30 == 0 and i != 0:
            current_value += monthly
            total_contributions += monthly
        current_value *= growth_factor / weighted.iloc[max(i - 1, 0)]
        contributions.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(total_contributions, 2)
        })
        portfolio_values.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": round(current_value, 2)
        })

    end_value = current_value
    starting_value = lump_sum + monthly * timeframe
    target_reached = end_value >= user_data.get("target_value", 0)
    portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

    timeline = {
        "contributions": contributions,
        "portfolio": portfolio_values
    }

    # Construct prompt for AI summary generation
    stocks_str = ", ".join([stock["symbol"] for stock in stocks_picked])
    goal = user_data.get("goal", "your investment goal")
    target_value = user_data.get("target_value", "N/A")
    timeframe_years = user_data.get("timeframe", "N/A")

    prompt = (
        f"The portfolio goal is '{goal}', targeting {target_value} over {timeframe_years} years. "
        f"The selected stocks are: {stocks_str}. "
        "Please provide an educational explanation about why these stocks might have been chosen, "
        "including insights into any volatile periods during this timeframe."
    )

    # Generate AI summary
    ai_summary = generate_ai_summary(prompt)

    # Save simulation to DB
    simulation = models.Simulation(
        user_id=sim_input.get("user_id"),
        name=goal,
        goal=goal,
        target_value=target_value,
        lump_sum=lump_sum,
        monthly=monthly,
        timeframe=timeframe,
        target_achieved=bool(target_reached),
        income_bracket=user_data.get("income_bracket"),
        risk_score=risk_score,
        risk_label=risk_label,
        ai_summary=ai_summary,
        results={
            "name": goal,
            "stocks_picked": stocks_picked,
            "starting_value": starting_value,
            "end_value": end_value,
            "return": round(portfolio_return, 2),
            "target_reached": bool(target_reached),
            "risk_score": risk_score,
            "risk_label": risk_label,
            "timeline": timeline
        }
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)

    return {
        "id": simulation.id,
        "user_id": simulation.user_id,
        "name": simulation.name,
        "goal": simulation.goal,
        "target_value": simulation.target_value,
        "lump_sum": simulation.lump_sum,
        "monthly": simulation.monthly,
        "timeframe": simulation.timeframe,
        "target_achieved": simulation.target_achieved,
        "income_bracket": simulation.income_bracket,
        "risk_score": simulation.risk_score,
        "risk_label": simulation.risk_label,
        "ai_summary": simulation.ai_summary,
        "results": simulation.results,
        "created_at": simulation.created_at.isoformat()
    }