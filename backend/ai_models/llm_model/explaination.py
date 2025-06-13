import torch

def generate_explanation(transaction, tokenizer, model, device=torch.device("cpu")):
    if tokenizer is None or model is None:
        return "Explanation service unavailable"

    try:
        text = (
            f"Amount: ${transaction['amount']:.2f}, "
            f"Credit Limit: ${transaction['credit_limit']:.2f}, "
            f"Ratio: {transaction['amount_ratio']:.2f}, "
            f"Chip Usage: {transaction.get('use_chip', 'N/A')}"
        )
        inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return f"Risk assessment: {probs[0][1]*100:.1f}% suspicious activity likelihood"
    except Exception as e:
        return f"Error generating explanation: {e}"