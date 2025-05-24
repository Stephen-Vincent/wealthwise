# backend/models/risk_analysis/predict_risk.py

import joblib
import pandas as pd

# Load the trained model
model = joblib.load("backend/models/risk_analysis/risk_model.pkl")

def predict_risk(user_input: dict) -> str:
    """
    Predict risk profile from user input dictionary.
    """
    df = pd.DataFrame([user_input])
    df_encoded = pd.get_dummies(df)

    # Ensure all expected columns are present
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Add missing columns with 0

    df_encoded = df_encoded[model_features]  # Reorder columns
    prediction = model.predict(df_encoded)
    return prediction[0]