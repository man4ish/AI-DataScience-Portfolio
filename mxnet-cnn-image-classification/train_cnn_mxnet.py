"""
This script demonstrates the process of building and training a simple Convolutional Neural Network (CNN) using MXNet. 
The network is applied to the CIFAR-10 dataset, a popular dataset for image classification.

The key steps involved include:
1. **Model Definition**: A simple CNN model is defined using MXNet's Gluon API. The model consists of two convolutional layers followed by fully connected layers, and it is designed for classifying CIFAR-10 images into one of 10 categories.
2. **Dataset Loading**: The CIFAR-10 dataset is loaded, and necessary transformations are applied, including normalization specific to CIFAR-10.
3. **Training Loop**: The model is trained for a set number of epochs using the Adam optimizer and Softmax cross-entropy loss. During training, the loss is calculated and used to update the model parameters.
4. **Model Evaluation**: After training, the model is evaluated on the test dataset, and accuracy is calculated.
5. **Visualization**: The script also includes a visualization step to display a sample image and its predicted label.

### Key Features:
- **GPU Support**: The script checks for the availability of a GPU and sets the context accordingly to speed up training.
- **Simple CNN**: The CNN architecture is simple, with two convolutional layers and fully connected layers, suitable for quick experimentation and learning.
- **DataLoader**: The `DataLoader` class is used to efficiently load batches of data during training and testing.
- **Evaluation**: The accuracy of the model is computed after training, allowing the user to assess the performance of the model.
- **Visualization**: Sample predictions are displayed using `matplotlib` to visually inspect the model's performance on a test image.

### Requirements:
- MXNet (installed with `pip install mxnet` or `pip install mxnet-cu102` for GPU support)
- Matplotlib (for visualization)

### Usage:
1. Ensure that the environment has the necessary libraries installed.
2. Run the script to train a simple CNN on the CIFAR-10 dataset.
3. The final accuracy and a visual output of predictions will be displayed at the end of training.
"""

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.data import DataLoader
import matplotlib.pyplot as plt

# Set the context for GPU or CPU
context = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()

# Define a simple CNN model
class SimpleCNN(nn.Block):
    def __init__(self, **kwargs):
        super(SimpleCNN, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(32, kernel_size=3, activation='relu')
        self.conv2 = nn.Conv2D(64, kernel_size=3, activation='relu')
        self.fc1 = nn.Dense(128, activation='relu')
        self.fc2 = nn.Dense(10)  # 10 classes for classification (CIFAR-10 dataset)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Load the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))  # CIFAR-10 specific normalization
])

train_dataset = datasets.CIFAR10(train=True).transform_first(transform)
test_dataset = datasets.CIFAR10(train=False).transform_first(transform)

train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
net = SimpleCNN()
net.initialize(ctx=context)

# Define loss and trainer
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.001})

# Train the model
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for data, label in train_data:
        data = data.as_in_context(context)
        label = label.as_in_context(context)
        
        with mx.autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        
        loss.backward()
        trainer.step(batch_size=data.shape[0])
        
        total_loss += loss.sum().asscalar()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")

# Evaluate the model
correct = 0
total = 0
for data, label in test_data:
    data = data.as_in_context(context)
    label = label.as_in_context(context)
    
    output = net(data)
    prediction = nd.argmax(output, axis=1)
    
    correct += (prediction == label).sum().asscalar()
    total += label.size
    
print(f"Accuracy: {correct / total * 100:.2f}%")

# Visualizing some predictions
data, label = test_dataset[0]
data = data.as_in_context(context)
output = net(data.expand_dims(axis=0))
prediction = nd.argmax(output, axis=1).asscalar()

plt.imshow(data.transpose((1, 2, 0)).asnumpy())
plt.title(f'Predicted Label: {prediction}')
plt.show()
