"""
This script demonstrates how to deploy the trained XGBoost model as a REST API using FastAPI.

Steps involved:
1. **FastAPI Setup**: Initializes a FastAPI app for serving predictions.
2. **Model Loading**: Loads the previously trained XGBoost model from the saved file.
3. **Prediction Endpoint**: Defines a POST endpoint that accepts input data, makes predictions, and returns the result.

Dependencies:
- fastapi
- uvicorn
- joblib
- numpy
"""

from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Step 1: Load the trained model
model = joblib.load("xgboost_model.pkl")

@app.post("/predict")
def predict(data: list):
    """
    Endpoint for making predictions using the trained XGBoost model.

    Args:
    - data: A list of feature values (e.g., a single data sample).

    Returns:
    - A dictionary with the predicted class label.
    """
    # Step 2: Convert the incoming data to a numpy array and reshape for prediction
    data = np.array(data).reshape(1, -1)
    
    # Step 3: Make prediction using the loaded model
    prediction = model.predict(data)
    
    # Step 4: Return the prediction as a response
    return {"prediction": int(prediction[0])}

# To run the FastAPI app, use:
# uvicorn script_name:app --reload
