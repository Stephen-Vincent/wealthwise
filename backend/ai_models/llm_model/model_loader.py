import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_llm_components(device=torch.device("cpu")):
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        print("LLM components loaded successfully!")
    except Exception as e:
        print(f"Failed to load LLM components: {e}")
        tokenizer, model = None, None
    return tokenizer, model