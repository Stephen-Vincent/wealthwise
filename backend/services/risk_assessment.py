def assess_risk_score(user_data):
    """
    Calculates a numeric risk score and returns a risk profile label.
    """
    score = 0

    # Score based on investment experience (0–60 pts)
    exp_years = min(int(user_data.get("experience", 0)), 30)
    score += exp_years * 2  # max 60 pts

    # Score based on self-assessed risk (0–30 pts)
    risk_pref = user_data.get("risk", "").lower()
    if risk_pref == "balanced":
        score += 15
    elif risk_pref == "adventurous":
        score += 30

    # Score based on timeframe (0–10 pts)
    timeframe = user_data.get("timeframe", "")
    if "5+" in timeframe:
        score += 10
    elif "1–5" in timeframe:
        score += 5

    # Return tier based on score
    if score < 25:
        risk_profile = "Cautious"
    elif score < 60:
        risk_profile = "Balanced"
    else:
        risk_profile = "Adventurous"

    print(f"[Risk Assessment] Computed risk score: {score}, Profile: {risk_profile}")
    return risk_profile, score