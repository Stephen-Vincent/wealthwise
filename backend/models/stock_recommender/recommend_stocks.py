import pandas as pd
import joblib
import numpy as np
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.database.models import OnboardingSubmission
from backend.models.stock_recommender.volatility_detector import calculate_volatility

# Load trained model and label binarizer
# Correct way to load model and label binarizer together
model, mlb = joblib.load("backend/models/stock_recommender/stock_model.pkl")

def recommend_stocks(user_data: dict, risk_score: int, db: Session, top_n=5):
    if not user_data:
        print("❌ No user data provided.")
        return []

    print(f"📥 User data received: {user_data}")
    print(f"🎯 Risk score: {risk_score}")

    # Build feature vector with expected structure
    feature_vector = pd.DataFrame([{
        "experience": user_data.get("experience", 0),
        "goal": user_data.get("goal", ""),
        "lump_sum": user_data.get("lump_sum", 0.0),
        "monthly": user_data.get("monthly", 0.0),
        "timeframe": user_data.get("timeframe", ""),
        "income_bracket": user_data.get("income_bracket", ""),
        "risk_score": risk_score
    }])

    print("🧠 Raw feature vector:")
    print(feature_vector)

    # Apply one-hot encoding
    feature_vector = pd.get_dummies(feature_vector)

    # Align columns with training set
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in feature_vector.columns:
            feature_vector[col] = 0
    feature_vector = feature_vector[expected_columns]

    # Predict probabilities for each stock
    y_pred = model.predict_proba(feature_vector)
    avg_proba = np.mean(np.vstack(y_pred), axis=0)

    print(f"🔍 Prediction probabilities: {avg_proba}")

    # Get top N recommended tickers
    top_indices = np.argsort(avg_proba)[-top_n:][::-1]
    recommended_tickers = mlb.classes_[top_indices]

    print(f"✅ Top {top_n} recommended tickers: {recommended_tickers}")

    # Optional: include volatility info
    recommendations = []
    for i, ticker_idx in enumerate(top_indices):
        ticker = mlb.classes_[ticker_idx]
        volatility = calculate_volatility(ticker)
        score = round(avg_proba[i], 4)
        print(f"📊 {ticker} - Volatility: {volatility}, Score: {score}")
        recommendations.append({
            "ticker": ticker,
            "volatility": round(volatility, 4),
            "score": score
        })

    return recommendations