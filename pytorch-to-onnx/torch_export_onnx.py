"""
pytorch_to_onnx.py

A complete example that demonstrates:
1. Defining a simple PyTorch model.
2. Exporting it to ONNX format.
3. Running inference using ONNX Runtime.

This script is useful for learning how to deploy PyTorch models 
outside the Python ecosystem using the ONNX format.

Dependencies:
- torch
- onnx
- onnxruntime
- numpy

Run:
    python pytorch_to_onnx.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import os

# Step 1: Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleModel()
model.eval()

# Step 2: Export the model to ONNX
onnx_filename = "simple_model.onnx"
dummy_input = torch.randn(1, 4)

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print(f"Model exported to {onnx_filename}")

# Step 3: Load and run the ONNX model using ONNX Runtime
session = ort.InferenceSession(onnx_filename)

# Create test input
test_input = np.random.randn(1, 4).astype(np.float32)
inputs = {"input": test_input}

# Run inference
outputs = session.run(["output"], inputs)

print("Input:", test_input)
print("Output:", outputs[0])
