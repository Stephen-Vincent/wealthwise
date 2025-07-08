# api/routers/ai_analysis.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services.portfolio_simulator import simulate_portfolio
from services.ai_analysis import AIAnalysisService
from database.session import get_db  # Fixed import

router = APIRouter(prefix="/ai", tags=["ai-analysis"])

# Initialize AI service
ai_service = AIAnalysisService()

@router.post("/simulate")
async def create_portfolio_simulation(sim_input: dict, db: Session = Depends(get_db)):
    """Create a new portfolio simulation with AI summary"""
    try:
        # Run the simulation (this calls your existing portfolio_simulator.py)
        result = simulate_portfolio(sim_input, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_portfolio(portfolio_data: dict):
    """Analyze existing portfolio performance"""
    try:
        result = await ai_service.analyze_portfolio_performance(portfolio_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-risk")
async def analyze_risk(portfolio_data: dict):
    """Analyze portfolio risk and allocation"""
    try:
        result = await ai_service.analyze_risk_allocation(portfolio_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain-changes")
async def explain_changes(portfolio_data: dict, previous_data: dict = None):
    """Explain portfolio changes over time"""
    try:
        result = await ai_service.explain_portfolio_changes(portfolio_data, previous_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))