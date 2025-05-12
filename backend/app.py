from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
from services.portfolio_simulator import simulate_portfolio as simulate_portfolio_logic
from database.models import OnboardingSubmission
from database.database import SessionLocal, engine

app = FastAPI(title="WealthWise Backend", version="1.0")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OnboardingRequest(BaseModel):
    name: str
    experience: int
    goal: str
    lumpSum: Optional[float] = 0.0
    monthly: Optional[float] = 0.0
    timeframe: str
    risk: str

class SimulationRequest(BaseModel):
    id: int

@app.get("/")
def read_root():
    return {"message": "WealthWise API is running."}

@app.post("/onboarding")
def save_onboarding(data: OnboardingRequest):
    try:
        db = SessionLocal()
        onboarding_data = OnboardingSubmission(
            name=data.name,
            experience=data.experience,
            goal=data.goal,
            lump_sum=data.lumpSum,
            monthly=data.monthly,
            timeframe=data.timeframe,
            risk=data.risk
        )
        db.add(onboarding_data)
        db.commit()
        db.refresh(onboarding_data)

        print("✅ Saved onboarding data:", {
            "id": onboarding_data.id,
            "name": onboarding_data.name,
            "experience": onboarding_data.experience,
            "goal": onboarding_data.goal,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "risk": onboarding_data.risk,
        })

        return {"id": onboarding_data.id}

    except Exception as e:
        print("❌ Error saving onboarding data:", e)
        raise HTTPException(status_code=500, detail="Failed to save onboarding data")

    finally:
        db.close()

@app.post("/simulate-portfolio")
def simulate_portfolio(data: SimulationRequest):
    try:
        db = SessionLocal()
        onboarding_data = db.query(OnboardingSubmission).filter(OnboardingSubmission.id == data.id).first()

        if onboarding_data is None:
            raise HTTPException(status_code=404, detail="Onboarding data not found")

        user_input = {
            "name": onboarding_data.name,
            "experience": onboarding_data.experience,
            "goal": onboarding_data.goal,
            "lump_sum": onboarding_data.lump_sum,
            "monthly": onboarding_data.monthly,
            "timeframe": onboarding_data.timeframe,
            "risk": onboarding_data.risk,
        }

        print("🚀 Simulating portfolio with input:", user_input)
        result = simulate_portfolio_logic(user_input)
        return result

    except Exception as e:
        print("❌ Error simulating portfolio:", e)
        raise HTTPException(status_code=500, detail="Portfolio simulation failed")

    finally:
        db.close()