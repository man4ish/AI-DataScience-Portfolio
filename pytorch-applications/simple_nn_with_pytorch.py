import torch
import torch.nn as nn
import torch.optim as optim

# === 1. Tensor Broadcasting ===
print("=== 1. Tensor Broadcasting ===")
a = torch.tensor([1.0, 2.0])  # Shape: (2,)
b = torch.tensor([1.0])       # Shape: (1,)

# Broadcasting allows us to add tensors of different shapes
c = a + b
print(f"Tensor a: {a}, shape: {a.shape}")
print(f"Tensor b: {b}, shape: {b.shape}")
print(f"Broadcasted result a + b: {c}, shape: {c.shape}\n")

# === 2. Building a Simple Neural Network ===
print("=== 2. Building a Simple Neural Network ===")
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Input layer: 2 input features, hidden layer with 4 neurons
        self.fc1 = nn.Linear(2, 4)  # Linear layer (2 input -> 4 output)
        self.fc2 = nn.Linear(4, 1)  # Output layer: 1 output

    def forward(self, x):
        # Apply ReLU activation function after first layer
        x = torch.relu(self.fc1(x))
        # Output layer (no activation function)
        x = self.fc2(x)
        return x

# Create a model instance
model = SimpleNN()
print(model)

# === 3. Defining a Loss Function & Optimizer ===
print("=== 3. Loss Function & Optimizer ===")
# Mean Squared Error Loss (for regression problems)
loss_fn = nn.MSELoss()

# Stochastic Gradient Descent (SGD) optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# === 4. Training Loop ===
print("=== 4. Training Loop ===")
# Example data (2 input features)
inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])  # Shape (3, 2)
# Example targets (expected outputs)
targets = torch.tensor([[1.0], [2.0], [3.0]])  # Shape (3, 1)

# Training loop (1 epoch for simplicity)
for epoch in range(1):  # Normally, you'd train for more epochs
    # Forward pass: Compute predicted y by passing inputs to the model
    predictions = model(inputs)
    
    # Compute the loss
    loss = loss_fn(predictions, targets)
    
    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()  # Reset gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    print(f"Epoch [{epoch+1}], Loss: {loss.item():.4f}")

# === 5. Model Prediction ===
print("\n=== 5. Model Prediction ===")
with torch.no_grad():  # No gradient calculation needed
    new_inputs = torch.tensor([[4.0, 5.0], [5.0, 6.0]])
    predictions = model(new_inputs)
    print(f"Predictions for new inputs: {predictions}\n")

