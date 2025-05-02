# Real-Time Image Classification with CNN

This project demonstrates real-time image classification using a pre-trained Convolutional Neural Network (CNN) model, specifically MobileNetV2, with TensorFlow and OpenCV. The model classifies images captured from a camera in real-time and provides predictions based on the ImageNet dataset.

## Features

- **Real-time Image Classification**: Classifies images from a live camera feed.
- **MobileNetV2 Model**: Uses a pre-trained MobileNetV2 model fine-tuned on the ImageNet dataset.
- **OpenCV Integration**: Captures live video from the webcam and processes each frame for classification.

## Requirements

- **Python 3.6+**
- **TensorFlow 2.x** (supports both CPU and GPU versions)
- **OpenCV**
- **NumPy**

You can install the required packages using `pip` or `conda`:

### Install via pip:
```bash
pip install tensorflow opencv-python numpy
```

## Install via conda:

```bash
pip install tensorflow opencv numpy
```

## Setup and Usage
Step 1: Clone the repository
Clone this repository to your local machine:


```bash
git clone <repository_url>
cd <repository_directory>
```

## Step 2: Download ImageNet Class Labels
This project uses the ImageNet class labels to map predictions to human-readable class names. You can download the imagenet_class_index.json file by running the following command:

```bash
curl -o imagenet_class_index.json https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
```

## Step 3: Run the Application
Once the setup is complete, you can run the image classification script using the following command:

```bash
python real_time_image_classification.py
```

This will launch a live camera feed, classify each frame, and display the predicted class label and confidence score.

## Step 4: Camera Access
Ensure that your system's camera is accessible and functional. If you face issues with camera initialization (e.g., permission errors or device not found), you may need to adjust the camera device index or check if another application is using the camera.

### Model Details
This project uses MobileNetV2, a lightweight and efficient CNN architecture, pre-trained on the ImageNet dataset. MobileNetV2 is optimized for mobile devices and can provide reasonable accuracy while maintaining a low computational cost.

### Troubleshooting
CUDA Error: If you're encountering errors related to CUDA (e.g., cuInit failure), ensure that your machine has a compatible GPU and the necessary NVIDIA drivers installed. If you're using a CPU-only machine, you can disable GPU support by setting the environment variable:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```
Camera Not Found: If the script cannot find the camera, ensure that it's correctly connected and that no other applications are using it. You can also try changing the camera device index in the code if necessary.