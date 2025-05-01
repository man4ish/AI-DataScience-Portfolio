from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

sequence = "M K T A Y I A K Q I K D L G"  # separate amino acids by space
tokens = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    embeddings = model(**tokens).last_hidden_state

print("🧬 Protein embedding shape:", embeddings.shape)  # [1, seq_len, hidden_dim]

