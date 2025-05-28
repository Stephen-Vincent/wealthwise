from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
from backend.services.portfolio_simulator import simulate_portfolio as simulate_portfolio_logic
from backend.database.models import OnboardingSubmission
from backend.database.database import SessionLocal, engine
import pandas as pd
import os
from backend.models.stock_recommender.volatility_detector import calculate_volatility
import joblib
import numpy as np
import json

app = FastAPI(title="WealthWise Backend", version="1.0")

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your React frontend
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
    consent: Optional[bool] = False

class SimulationRequest(BaseModel):
    id: int

@app.get("/")
def read_root():
    return {"message": "WealthWise API is running."}

@app.post("/onboarding")
def save_onboarding(data: OnboardingRequest):
    try:
        print(f"📥 Received onboarding data: {data}")
        # Estimate target value if needed (or remove entirely if not used at onboarding)
        print("🎯 Preparing to assess risk and save onboarding data.")
        from backend.services.risk_assessment import assess_risk_score
        computed_risk, risk_score = assess_risk_score(data.dict())
        db = SessionLocal()
        onboarding_data = OnboardingSubmission(
            name=data.name,
            experience=data.experience,
            goal=data.goal,
            lump_sum=data.lumpSum,
            monthly=data.monthly,
            timeframe=data.timeframe,
            risk=computed_risk,
            risk_score=risk_score
        )
        db.add(onboarding_data)
        db.commit()
        db.refresh(onboarding_data)

        print(f"✅ Saved onboarding data to DB: {onboarding_data}")

        return {"id": onboarding_data.id, "risk": computed_risk, "risk_score": risk_score}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save onboarding data")

    finally:
        db.close()


# ---- Stock Name Map Endpoint ----
@app.get("/stock-name-map")
def get_stock_name_map():
    try:
        file_path = "backend/models/stock_recommender/stock_name_map.json"
        with open(file_path, "r") as f:
            raw_data = json.load(f)

        # Convert from: { "ADBE": "ADBE" } to { "ADBE": { "name": "Adobe Inc." } }
        stock_name_map = {
            ticker: {"name": name} for ticker, name in raw_data.items()
        }

        return stock_name_map

    except Exception as e:
        print(f"❌ Failed to load stock name map: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load stock name map")

@app.post("/simulate-portfolio")
def simulate_portfolio(data: SimulationRequest):
    try:
        print(f"🔍 Fetching onboarding data for ID: {data.id}")
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
            "selected_stocks": None,
        }
        print(f"🧠 User input for simulation: {user_input}")

        result = simulate_portfolio_logic(user_input)
        result["risk"] = onboarding_data.risk  
        result["target_value"] = onboarding_data.target_value
        
        return result

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Portfolio simulation failed")

    finally:
        db.close()


# ---- Stock Recommendation Endpoint ----
class RecommendationRequest(BaseModel):
    risk_score: int
    timeframe: int

@app.post("/recommend-stocks")
def recommend_stocks(data: RecommendationRequest):
    try:
        print(f"📥 Received recommendation input: {data}")
        # Load model and label binarizer
        model = joblib.load("backend/models/stock_recommender/stock_model.pkl")
        mlb = joblib.load("backend/models/stock_recommender/label_binarizer.pkl")

        # Format input for model
        features = np.array([[data.risk_score, data.timeframe]])

        predictions = model.predict(features)
        predicted_labels = mlb.inverse_transform(predictions)[0]

        recommendations = []
        for ticker in predicted_labels:
            vol = calculate_volatility(ticker)
            recommendations.append({
                "ticker": ticker,
                "volatility": round(vol, 4) if vol else None
            })

        return {"recommendations": recommendations}

    except Exception as e:
        print(f"❌ Recommendation Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate stock recommendations")

@app.delete("/clear-database")
def clear_database():
    try:
        db = SessionLocal()
        deleted = db.query(OnboardingSubmission).delete()
        db.commit()
        print(f"🗑️ Cleared {deleted} onboarding records.")
        return {"message": f"Deleted {deleted} records."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear database")
    finally:
        db.close()
