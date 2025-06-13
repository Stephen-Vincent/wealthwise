from ai_models.llm_model.financial_chat import financial_chat
import os
from ai_models.llm_model.groq_client import get_groq_client

# Load GROQ_API_KEY from environment variables
try:
    groq_client = get_groq_client()
except ValueError as e:
    print(f"Warning: {e}")
    groq_client = None

def generate_ai_summary(prompt: str) -> str:
    if groq_client is None:
        # Return a hardcoded placeholder summary if no API key is set
        return "AI summary unavailable - API key missing."

    try:
        response = financial_chat(prompt, groq_client=groq_client)
        return response
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return "Sorry, I was unable to generate the summary at this time."

def summarize_portfolio(simulation: dict) -> str:
    results = simulation.get("results", {})
    stocks = results.get("stocks_picked", [])

    if not simulation or not isinstance(stocks, list) or not stocks:
        return "⚠️ No portfolio data available to summarize."

    goal = simulation.get("goal", "No goal specified")
    lump_sum = simulation.get("lump_sum", 0)
    monthly = simulation.get("monthly", 0)
    timeframe = simulation.get("timeframe", "N/A")
    target_achieved = simulation.get("target_achieved", False)
    risk_label = simulation.get("risk_label", "N/A")
    start_value = results.get("starting_value", "N/A")
    end_value = results.get("end_value", "N/A")

    stock_list = ", ".join(
        stock.get("symbol", "UNKNOWN") if isinstance(stock, dict) else "UNKNOWN"
        for stock in stocks
    )

    def format_currency(value):
        return f"£{value:,.2f}" if isinstance(value, (int, float)) else value

    start_value_str = format_currency(start_value)
    end_value_str = format_currency(end_value)

    ai_prompt = (
    "Act as a knowledgeable financial advisor and educator. "
    f"A user invested an initial amount of £{lump_sum:,.2f} with monthly contributions of £{monthly:,.2f} "
    f"over a period of {timeframe} years, choosing a {risk_label.lower()} risk portfolio. "
    f"The portfolio began with a value of £{start_value:,.2f} and grew to £{end_value:,.2f} by the end of the period. "
    f"The investment was allocated among the following stocks: {stock_list}. "
    "Please explain in clear, educational terms what happened over the investment period, including key factors influencing portfolio growth, "
    "the role of contributions and compounding, risks faced, and how the selected stocks contributed to the overall performance. "
    "Focus on helping the user understand the investment process, market dynamics, and what these results mean for their financial goals."
)

    ai_summary = generate_ai_summary(ai_prompt)

    summary = {ai_summary}
    

    print(f"[DEBUG] AI Summary Generated: {summary}")
    return summary