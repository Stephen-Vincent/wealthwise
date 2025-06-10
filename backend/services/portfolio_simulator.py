from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.orm import Session
from database import models
import yfinance as yf
import numpy as np
import pandas as pd

def simulate_portfolio(sim_input: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Simulate a user's portfolio using the provided onboarding input.
    
    This function receives onboarding data from the API layer, which already includes:
    - User's investment preferences and financial information.
    - A pre-calculated `risk_score` and `risk_label` returned from the risk model.

    It uses these inputs to:
    - Choose a hardcoded set of stocks (this will be replaced by an AI recommender later).
    - Simulate portfolio growth over time (e.g. a 50% return).
    - Determine whether the user's target goal is reached.
    - Return data that will be passed to the PortfolioContext and then displayed in the frontend dashboard.
    """
    
    # ✅ Extract relevant input fields (used for simulation and stock selection logic)
    user_data = {
        "experience": sim_input.get("experience"),              # Not used yet, but available for future logic
        "goal": sim_input.get("goal"),                          # Used for naming portfolio
        "target_value": sim_input.get("target_value"),
        "lump_sum": sim_input.get("lump_sum"),
        "monthly": sim_input.get("monthly"),
        "timeframe": sim_input.get("timeframe"),
        "income_bracket": sim_input.get("income_bracket")       # Also not yet used
    }

    # ✅ Extract risk values already calculated from risk_assessor.py
    risk_score = sim_input.get("risk_score")                   # e.g. 3
    risk_label = sim_input.get("risk_label")                   # e.g. 'moderate'

    lump_sum = float(sim_input.get("lump_sum") or 0)
    monthly = float(sim_input.get("monthly") or 0)

    # Calculate start_date and end_date based on timeframe
    timeframe = int(sim_input.get("timeframe") or 0)
    today = datetime.today()
    if timeframe <= 1:
        start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
    elif timeframe <= 5:
        start_date = (today - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
    else:
        start_date = (today - timedelta(days=10 * 365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    # ✅ Select stocks based on risk_label
    if risk_label and risk_label.lower() == "conservative":
        allocation = round(1 / 5, 2)
        stocks_picked = [
            {"symbol": "JNJ", "name": "Johnson & Johnson", "allocation": allocation},
            {"symbol": "PG", "name": "Procter & Gamble", "allocation": allocation},
            {"symbol": "KO", "name": "Coca-Cola Co", "allocation": allocation},
            {"symbol": "PEP", "name": "PepsiCo Inc", "allocation": allocation},
            {"symbol": "VZ", "name": "Verizon Communications", "allocation": allocation}
        ]
    elif risk_label and risk_label.lower() == "aggressive":
        allocation = round(1 / 5, 2)
        stocks_picked = [
            {"symbol": "TSLA", "name": "Tesla Inc.", "allocation": allocation},
            {"symbol": "NVDA", "name": "NVIDIA Corp.", "allocation": allocation},
            {"symbol": "AMD", "name": "Advanced Micro Devices", "allocation": allocation},
            {"symbol": "SQ", "name": "Block Inc.", "allocation": allocation},
            {"symbol": "SHOP", "name": "Shopify Inc.", "allocation": allocation}
        ]
    else:  # moderate or unknown
        allocation = round(1 / 5, 2)
        stocks_picked = [
            {"symbol": "AAPL", "name": "Apple Inc.", "allocation": allocation},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "allocation": allocation},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "allocation": allocation},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "allocation": allocation},
            {"symbol": "V", "name": "Visa Inc.", "allocation": allocation}
        ]

    # ✅ Calculate starting investment amount
    # Formula: lump sum + (monthly contribution × number of years)
    starting_value = lump_sum + monthly * timeframe

    # --- Use yfinance to simulate actual portfolio growth ---
    tickers = [stock["symbol"] for stock in stocks_picked]  # extract ticker strings
    start_date = (datetime.today() - timedelta(days=timeframe * 365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # Calculate equal weights for simplicity
    weights = np.ones(len(tickers)) / len(tickers)
    # Calculate normalized returns
    normalized = data / data.iloc[0]
    weighted = normalized.dot(weights)
    # Calculate daily portfolio value over time with compounding
    portfolio_values = []
    contributions = []
    current_value = lump_sum
    total_contributions = lump_sum
    for i, (date, growth_factor) in enumerate(weighted.items()):
        if i % 30 == 0 and i != 0:  # approx. monthly
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
    # ✅ Determine if the goal was achieved
    target_reached = end_value >= sim_input.get("target_value", 0)
    # ✅ Calculate total return as a decimal (e.g., 0.5 = 50%)
    portfolio_return = (end_value - starting_value) / starting_value if starting_value > 0 else 0

    target_reached = bool(target_reached)
    end_value = float(end_value)
    starting_value = float(starting_value)
    portfolio_return = float(round(portfolio_return, 2))

    timeline = {
        "contributions": contributions,
        "portfolio": portfolio_values
    }

    # Convert portfolio timeline values
    for point in portfolio_values:
        point["value"] = float(point["value"])
    for point in contributions:
        point["value"] = float(point["value"])

    from .summary_generator import summarize_portfolio
    ai_summary = summarize_portfolio({
        "name": sim_input.get("goal"),
        "goal": sim_input.get("goal"),
        "target_value": sim_input.get("target_value"),
        "lump_sum": lump_sum,
        "monthly": monthly,
        "timeframe": timeframe,
        "income_bracket": sim_input.get("income_bracket"),
        "risk_score": risk_score,
        "risk_label": risk_label,
        "results": {
            "stocks_picked": stocks_picked,
            "starting_value": starting_value,
            "end_value": end_value,
            "return": portfolio_return,
            "target_reached": target_reached,
            "timeline": timeline
        }
    })

   

    # ✅ Save simulation result to the database
    simulation = models.Simulation(
        user_id=sim_input.get("user_id"),
        name=sim_input.get("goal"),
        goal=sim_input.get("goal"),
        target_value=sim_input.get("target_value"),
        lump_sum=sim_input.get("lump_sum"),
        monthly=sim_input.get("monthly"),
        timeframe=sim_input.get("timeframe"),
        target_achieved=target_reached,
        income_bracket=sim_input.get("income_bracket"),
        risk_score=risk_score,
        risk_label=risk_label,
        ai_summary=ai_summary,
        results={
            "name": sim_input.get("goal"),
            "stocks_picked": stocks_picked,
            "starting_value": starting_value,
            "end_value": end_value,
            "return": portfolio_return,
            "target_reached": target_reached,
            "risk_score": risk_score,
            "risk_label": risk_label,
            "timeline": timeline
        }
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)


    print("Simulation result to return:", {
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
        "results": simulation.results,
        "created_at": simulation.created_at.isoformat()
    })
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
        "results": simulation.results,
        "created_at": simulation.created_at.isoformat()
    }