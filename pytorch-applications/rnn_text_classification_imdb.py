import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# === 1. Load and Tokenize Dataset ===
print("=== 1. Loading IMDb Dataset ===")
train_iter, test_iter = IMDB(split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

# Reload iterator (it gets exhausted)
train_iter, test_iter = IMDB(split=('train', 'test'))

# === 2. Data Pipeline ===
def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(label): return 1 if label == 'pos' else 0

def collate_batch(batch):
    text_list, label_list = [], []
    for label, text in batch:
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(torch.tensor(label_pipeline(label), dtype=torch.float))
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return text_list, torch.tensor(label_list)

train_loader = DataLoader(list(train_iter), batch_size=8, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=8, shuffle=False, collate_fn=collate_batch)

# === 3. Define the RNN Model ===
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.sigmoid(self.fc(hidden.squeeze(0)))

# === 4. Model, Loss, Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextRNN(len(vocab), 64, 128, 1).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 5. Training Loop ===
print("\n=== 5. Training the RNN ===")
for epoch in range(3):
    model.train()
    total_loss = 0
    for text_batch, label_batch in train_loader:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        output = model(text_batch).squeeze()
        loss = loss_fn(output, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# === 6. Evaluation ===
print("\n=== 6. Evaluating on Test Set ===")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for text_batch, label_batch in test_loader:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        output = model(text_batch).squeeze()
        predictions = (output >= 0.5).float()
        correct += (predictions == label_batch).sum().item()
        total += label_batch.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")
