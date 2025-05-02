import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# === 1. Load IMDb Dataset ===
print("Loading IMDb dataset...")
train_iter, test_iter = IMDB(split=('train', 'test'))
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
train_iter, test_iter = IMDB(split=('train', 'test'))  # reset iterator

# === 2. Pipelines ===
def text_pipeline(text):
    return vocab(tokenizer(text))

def label_pipeline(label):
    return 1 if label == 'pos' else 0

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

# === 3. Define LSTM Model ===
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.sigmoid(self.fc(hidden[-1]))

# === 4. Training Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextLSTM(len(vocab), embed_dim=64, hidden_dim=128, output_dim=1).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 5. Training Loop ===
print("Training LSTM...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for text_batch, label_batch in train_loader:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        predictions = model(text_batch).squeeze()
        loss = loss_fn(predictions, label_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

# === 6. Evaluation ===
print("Evaluating on test set...")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for text_batch, label_batch in test_loader:
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        predictions = model(text_batch).squeeze()
        predicted_labels = (predictions >= 0.5).float()
        correct += (predicted_labels == label_batch).sum().item()
        total += label_batch.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")
