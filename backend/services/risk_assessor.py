# services/risk_assessor.py

"""
This module evaluates user risk based on onboarding input using a trained model.
It returns:
- A numerical risk score
- A corresponding risk label
"""

import joblib
import os
import pandas as pd

# Define paths to the trained model and encoders
# These files were generated during the training phase and saved for inference
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, '../ai_models/risk_model/model.pkl')
# ENCODER_PATH = os.path.join(BASE_DIR, '../ai_models/risk_model/encoders.pkl')

# Load the trained model and encoders into memory for prediction use
# model = joblib.load(MODEL_PATH)
# encoders = joblib.load(ENCODER_PATH)

def get_risk_label(score: int) -> str:
    """
    Converts a numerical risk score into a human-readable risk label.
    Used after the model returns the risk score.
    
    Args:
        score (int): Risk score predicted by the model

    Returns:
        str: Corresponding risk label ("Conservative", "Moderate", or "Aggressive")
    """
    if score < 35:
        return "Conservative"
    elif score < 65:
        return "Moderate"
    else:
        return "Aggressive"

def calculate_user_risk(sim_input) -> dict:
    """
    Simplified placeholder version of the risk assessor.
    This will be replaced with AI model logic later.
    """
    # If sim_input is a Pydantic model, convert it to a dictionary
    if hasattr(sim_input, "dict"):
        sim_input = sim_input.dict()

    print(f"[DEBUG] Placeholder input received: {sim_input}")

    # Just extract the values (no transformation)
    features = [
        sim_input.get("years_of_experience"),
        sim_input.get("goal"),
        sim_input.get("target_value"),
        sim_input.get("lump_sum"),
        sim_input.get("monthly"),
        sim_input.get("timeframe"),
        sim_input.get("income_bracket")
    ]

    print(f"[DEBUG] Features: {features}")

    risk_score = 50
    risk_label = "Moderate"
    return risk_score, risk_label