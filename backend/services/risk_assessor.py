# services/risk_assessor.py

"""
This module evaluates user risk based on onboarding input using a trained model.
It returns:
- A numerical risk score
- A corresponding risk label
"""

import pandas as pd

def calculate_user_risk(sim_input) -> dict:
    """
    Temporarily bypass model prediction and return a hardcoded risk score and label.

    Args:
        sim_input: User input data, can be a dict or a Pydantic model with .dict() method.

    Returns:
        tuple: Hardcoded risk score and corresponding risk label.
    """
    # Convert Pydantic model to dictionary if needed
    if hasattr(sim_input, "dict"):
        sim_input = sim_input.dict()

    # Debug input received
    print(f"[DEBUG] Input received (risk_assessor bypass): {sim_input}")

    # Hardcoded values for now
    risk_score = 40  # e.g., Moderate risk score
    risk_label = "Medium"  # Corresponding risk label

    return risk_score, risk_label