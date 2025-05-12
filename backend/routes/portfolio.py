from fastapi import APIRouter
from pydantic import BaseModel
from services.portfolio_simulator import simulate_portfolio

router = APIRouter()

class PortfolioRequest(BaseModel):
    risk: str = "Balanced"
    lump_sum: float = 0
    monthly: float = 0
    start_date: str
    end_date: str

@router.post("/simulate")
def simulate(request: PortfolioRequest):
    print("Received portfolio simulation request:", request.dict())
    result = simulate_portfolio(request.dict())
    return result
