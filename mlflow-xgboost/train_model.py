"""
This script demonstrates an end-to-end AI pipeline using XGBoost for training, MLflow for experiment tracking, and model deployment.

Steps involved:
1. **Data Ingestion**: Loads the Iris dataset.
2. **Data Preprocessing**: Splits the dataset into training and testing sets.
3. **Model Training**: Trains an XGBoost model with specified hyperparameters and logs the parameters with MLflow.
4. **Model Evaluation**: Evaluates the model on the test set and logs performance metrics to MLflow.
5. **Model Deployment**: Saves the trained model using `joblib` and logs it to MLflow.
6. **Monitoring and Logging**: Logs metrics, artifacts, and prints model performance.

Dependencies:
- mlflow
- xgboost
- numpy
- pandas
- scikit-learn
- joblib

This script can be expanded for use in real-world AI/ML production pipelines.
"""

import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Data Ingestion - Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Step 2: Data Preprocessing - Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training - Set up MLflow tracking
mlflow.start_run()

# Log hyperparameters
params = {
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100
}

mlflow.log_params(params)

# Initialize and train an XGBoost model
model = xgb.XGBClassifier(
    objective=params["objective"],
    num_class=params["num_class"],
    max_depth=params["max_depth"],
    learning_rate=params["learning_rate"],
    n_estimators=params["n_estimators"]
)

model.fit(X_train, y_train)

# Step 4: Model Evaluation - Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log evaluation metrics to MLflow
mlflow.log_metric("accuracy", accuracy)

# Step 5: Model Deployment - Save the trained model
model_path = "xgboost_model.pkl"
joblib.dump(model, model_path)

# Log the trained model to MLflow
mlflow.log_artifact(model_path, "xgboost_model")

# End the MLflow run
mlflow.end_run()

# Step 6: Monitoring and Logging - Print model performance
print(f"Model accuracy: {accuracy:.4f}")
