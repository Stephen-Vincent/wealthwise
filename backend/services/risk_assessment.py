
from services.stock_selector import get_stocks_for_risk_profile as select_stocks

def assess_risk_score(user_data):
    """
    Calculates a risk score based on user data.
    """
    score = 0

    # Score based on investment experience
    if user_data.get("experience", 0) >= 5:
        score += 2
    elif user_data.get("experience", 0) >= 2:
        score += 1

    # Score based on financial goals
    goal = user_data.get("goal", "").lower()
    if "retire" in goal or "house" in goal:
        score += 1
    if "millionaire" in goal or "growth" in goal:
        score += 2

    # Score based on investment timeframe
    timeframe = user_data.get("timeframe", "")
    if "5+" in timeframe:
        score += 2
    elif "1–5" in timeframe:
        score += 1

    # Convert score to risk profile
    if score >= 5:
        return "Adventurous"
    elif score >= 3:
        return "Balanced"
    else:
        return "Cautious"

def get_stocks_based_on_user_data(user_data):
    """
    Returns appropriate stocks after assessing the user's risk profile.
    """
    risk_profile = assess_risk_score(user_data)
    return select_stocks(risk_profile), risk_profile