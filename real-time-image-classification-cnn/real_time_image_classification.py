import cv2
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Define the list of ImageNet class labels
class_labels = {i: label for i, label in enumerate(open('imagenet_class_index.json'))}

def preprocess_image(image):
    """
    Preprocess the image to fit the input requirements of MobileNetV2.
    
    Parameters:
    image (ndarray): The image to be preprocessed.
    
    Returns:
    ndarray: Preprocessed image ready for MobileNetV2 input.
    """
    image_resized = cv2.resize(image, (224, 224))  # Resize image to 224x224
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_array = np.expand_dims(image_rgb, axis=0)  # Add batch dimension
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

def classify_frame(frame):
    """
    Classify the given frame using the pre-trained MobileNetV2 model.
    
    Parameters:
    frame (ndarray): The frame captured from the webcam to classify.
    
    Returns:
    str: The predicted class label.
    """
    preprocessed_image = preprocess_image(frame)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]  # Return the top predicted label

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Classify the current frame
    predicted_label = classify_frame(frame)
    
    # Display the prediction on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Real-Time Image Classification', frame)
    
    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
