import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === 1. Training for Multiple Epochs ===
print("=== 1. Training for Multiple Epochs ===")
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # Input layer to hidden layer (2 -> 4)
        self.fc2 = nn.Linear(4, 1)  # Hidden layer to output layer (4 -> 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation function after first layer
        x = self.fc2(x)  # Output layer (no activation function)
        return x

# Example data (2 input features)
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
# Example targets (expected outputs)
targets = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

# Dataset and DataLoader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model, loss function, and optimizer
model = SimpleNN()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop for multiple epochs
epochs = 5
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()  # Reset gradients
        predictions = model(batch_inputs)  # Forward pass
        loss = loss_fn(predictions, batch_targets)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()  # Accumulate loss

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# === 2. Model Evaluation ===
print("\n=== 2. Model Evaluation ===")
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to track gradients during evaluation
    test_inputs = torch.tensor([[5.0, 6.0], [6.0, 7.0]])
    predictions = model(test_inputs)
    print(f"Predictions for new inputs: {predictions}")

# === 3. Saving and Loading the Model ===
print("\n=== 3. Saving and Loading the Model ===")
# Save the model's state_dict (weights)
torch.save(model.state_dict(), 'simple_nn.pth')
print("Model saved successfully!")

# Load the model (for future use)
model_loaded = SimpleNN()
model_loaded.load_state_dict(torch.load('simple_nn.pth'))
print("Model loaded successfully!")

# Test the loaded model
model_loaded.eval()
with torch.no_grad():
    predictions = model_loaded(test_inputs)
    print(f"Predictions from the loaded model: {predictions}")

