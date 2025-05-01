# PyTorch to ONNX Example

This repository demonstrates how to:

1. Define a simple PyTorch model.
2. Export it to ONNX format.
3. Load and run the exported model using ONNX Runtime.

This script provides a basic walkthrough of converting a PyTorch model to ONNX and performing inference using ONNX Runtime, making it easier to deploy models in non-Python environments.

---

### Files

- `pytorch_to_onnx.py` – The main script that defines the model, exports it to ONNX, and runs inference.
- `simple_model.onnx` – The ONNX model generated after running the script.

---

### Installation

To run this example, you will need the following Python libraries:

```bash
pip install torch torchvision onnx onnxruntime numpy
```
How to Run
Clone the repository (if applicable):

```bash
git clone https://github.com/your-username/pytorch-to-onnx.git
cd pytorch-to-onnx
```

### Run the script:

```bash
python pytorch_to_onnx.py
```

This will:

- Define a simple neural network in PyTorch.

- Export the model to the simple_model.onnx file.

- Load the model using ONNX Runtime.

### Run inference and print the results.

### Example Output
When you run the script, you should see something like the following:

```
Model exported to simple_model.onnx
Input: [[ 0.325  -0.754   0.189   1.120 ]]
Output: [[ 0.038 -0.105]]
```

(Note: The values will vary depending on the random input generated for inference.)

# Converting ONNX Model to TensorFlow

### Convert the ONNX Model to TensorFlow:
We use the onnx-tf package to convert the ONNX model to TensorFlow format. The model is saved in the tensorflow_model directory in SavedModel format.

### Run TensorFlow Inference:
After the conversion, the model is loaded in TensorFlow, and we run inference to ensure that the conversion was successful.

