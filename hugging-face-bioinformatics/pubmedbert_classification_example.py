from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Use a publicly available PubMedBERT model
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example biomedical sentence
text = "Aspirin is used to reduce fever and relieve mild to moderate pain."

# Tokenize and encode
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Forward pass
with torch.no_grad():
    logits = model(**inputs).logits

# Convert logits to probabilities
probs = F.softmax(logits, dim=1)

# Print results
print("🧬 Input Text:", text)
print("🔢 Probabilities:", probs.tolist())
print("📊 Predicted Class:", torch.argmax(probs).item())

