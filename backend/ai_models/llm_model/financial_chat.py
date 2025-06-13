from .groq_client import get_groq_client

def financial_chat(user_input, groq_client=None):
    if groq_client is None:
        try:
            groq_client = get_groq_client()
        except Exception as e:
            return f"Error initializing chat service: {e}"

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are FinanceWise AI, a strict financial advisor. Only respond to finance-related questions. "
                        "If the user asks about anything unrelated to finance (e.g. languages, science, entertainment), reply: "
                        "'Sorry, I can only help with finance-related questions.'"
                    ),
                },
                {"role": "user", "content": user_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            top_p=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during chat: {e}"