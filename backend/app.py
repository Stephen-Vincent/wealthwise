from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
from services.portfolio_simulator import simulate_portfolio as simulate_portfolio_logic

app = FastAPI(title="WealthWise Backend", version="1.0")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data model from frontend
class SimulationRequest(BaseModel):
    name: str
    experience: int
    goal: str
    lumpSum: Optional[float] = 0.0
    monthly: Optional[float] = 0.0
    timeframe: str
    risk: str

@app.get("/")
def read_root():
    return {"message": "WealthWise API is running."}

@app.post("/simulate-portfolio")
def simulate_portfolio(data: SimulationRequest):
    result = simulate_portfolio_logic({
        "name": data.name,
        "experience": data.experience,
        "goal": data.goal,
        "lump_sum": data.lumpSum,
        "monthly": data.monthly,
        "timeframe": data.timeframe,
        "risk": data.risk,
    })
    return result