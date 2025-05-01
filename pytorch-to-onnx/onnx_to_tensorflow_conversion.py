import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load the ONNX model
onnx_model_path = "simple_model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Save the converted model in TensorFlow format (SavedModel)
tf_rep.export_graph("tensorflow_model")

print("Model successfully converted to TensorFlow format!")

# Optional: Test the TensorFlow model
# Load the TensorFlow SavedModel
saved_model = tf.saved_model.load("tensorflow_model")

# Get the inference function
infer = saved_model.signatures["serving_default"]

# Prepare input for inference
import numpy as np
test_input = np.random.randn(1, 4).astype(np.float32)

# Run inference
output = infer(tf.convert_to_tensor(test_input))

# Print output
print("TensorFlow Inference Output:", output)
