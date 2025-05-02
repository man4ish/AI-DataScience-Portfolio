import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# === 1. Data Loading and Preprocessing ===
print("=== 1. Data Loading and Preprocessing ===")

# Define the transformation to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Download the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for batch processing
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# === 2. Building the CNN Model ===
print("\n=== 2. Building the CNN Model ===")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # 3 input channels (RGB), 32 output channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Flatten the output of conv layers (assuming 32x32 input)
        self.fc2 = nn.Linear(512, 10)  # 10 output classes (CIFAR-10 has 10 categories)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply first conv layer + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Apply second conv layer + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))  # Apply third conv layer + ReLU + Pooling
        
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor for fully connected layer
        
        x = torch.relu(self.fc1(x))  # Apply fully connected layer + ReLU
        x = self.fc2(x)  # Output layer (no activation function)
        return x

# Instantiate the model
model = CNN()
print(model)

# === 3. Loss Function and Optimizer ===
print("\n=== 3. Loss Function and Optimizer ===")
loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 4. Training Loop ===
print("\n=== 4. Training Loop ===")
epochs = 5
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)  # Calculate the loss

        # Backward pass
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()  # Accumulate loss
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

# === 5. Model Evaluation ===
print("\n=== 5. Model Evaluation ===")
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():  # No gradient calculation during evaluation
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%")

# === 6. Saving the Model ===
print("\n=== 6. Saving the Model ===")
torch.save(model.state_dict(), 'cnn_cifar10.pth')
print("Model saved successfully!")

