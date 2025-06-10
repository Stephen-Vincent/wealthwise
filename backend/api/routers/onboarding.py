from datetime import datetime
from fastapi import APIRouter, status
from sqlalchemy.orm import Session
from fastapi import Depends

from database import schemas
from database.session import get_db
from services.risk_assessor import calculate_user_risk
from services.portfolio_simulator import simulate_portfolio
from services.summary_generator import summarize_portfolio

router = APIRouter()

"""
Endpoint: POST /onboarding/

Expected input (schemas.OnboardingCreate):
{
    "years_of_experience": int,       # e.g., 4
    "goal": str,                      # e.g., "retirement" — must be one of: ['growth', 'income', 'preservation']
    "target_value": float,            # e.g., 50000.0
    "lump_sum": float,                # e.g., 3000.0
    "monthly": float,                 # e.g., 250.0
    "timeframe": int,                 # e.g., 5
    "income_bracket": str,           # e.g., "medium"
    "consent": bool,                  # e.g., True
    "name": str,                      # e.g., "Stephen Vincent"
    "user_id": int                    # e.g., 1 (Not used internally in risk assessment or simulation)
}

Workflow:
1. Accept onboarding data from frontend.
2. Pass it to `calculate_user_risk()` to get a risk score and label.
3. Merge this risk data with the original input.
4. Pass the merged input to `simulate_portfolio()` to generate portfolio results.
5. Return the result to frontend in the shape of `schemas.SimulationResponse`.
"""

@router.post("/", status_code=status.HTTP_201_CREATED)
def create_onboarding(onboarding_data: schemas.OnboardingCreate, db: Session = Depends(get_db)):
    # Step 1: Calculate risk score and label from onboarding input
    risk_score, risk_label = calculate_user_risk(onboarding_data)

    # Step 2: Merge risk data with onboarding input
    simulation_input = onboarding_data.dict()
    simulation_input["risk_score"] = risk_score
    simulation_input["risk_label"] = risk_label

    # Step 3: Run portfolio simulation with full input
    simulation_result = simulate_portfolio(simulation_input, db)
    print("🔍 Simulation Result:", simulation_result)

    # Step 4: Generate AI summary of the portfolio simulation
    ai_summary = summarize_portfolio(simulation_result)

    # Step 4.5: Store AI summary in simulation result for persistence
    simulation_result["ai_summary"] = ai_summary

    # Step 5: Construct response payload
    return {
        "id": simulation_result["id"],
        "user_id": onboarding_data.user_id,
        "name": onboarding_data.name,
        "goal": onboarding_data.goal,
        "target_value": onboarding_data.target_value,
        "lump_sum": onboarding_data.lump_sum,
        "monthly": onboarding_data.monthly,
        "timeframe": onboarding_data.timeframe,
        "target_achieved": simulation_result["results"]["target_reached"],
        "income_bracket": onboarding_data.income_bracket,
        "risk_score": risk_score,
        "risk_label": risk_label,
        **simulation_result,
        "created_at": datetime.utcnow().isoformat()
    }