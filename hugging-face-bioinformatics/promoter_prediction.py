from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Set model name
model_name = "zhihan1996/DNABERT-2-117M"

# Load config FIRST using trust_remote_code
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

# Load tokenizer and model with matching config and trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=True)
model.eval()

# Example DNA sequence (at least 101bp is good for DNABERT)
sequence = "ACGTGACCTGAGGCGTGTGGCTTTTTTTAGAGGGCCCTCGCTGAGACCTAGCGTCGATCGGTCGACCTAGCTAGTCAGCTAGTCAGCTAGT"

# Tokenize
inputs = tokenizer(sequence, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=1).item()

# Display result
labels = ["Non-Promoter", "Promoter"]
print(f"\n🧬 Prediction: {labels[predicted_class]}")
print(f"🔢 Probabilities: {probs.squeeze().tolist()}")

