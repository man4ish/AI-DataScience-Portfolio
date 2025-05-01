# MLflow XGBoost Project

This repository demonstrates how to use MLflow for tracking machine learning experiments and deploying an XGBoost model. It includes the training, logging, and deployment of a model with XGBoost, and the integration of MLflow for experiment tracking.

## Step 1: Clone the Repository

If you haven’t already, clone the project repository to your local machine by running the following command in your terminal:

```bash
git clone https://github.com/yourusername/mlflow-xgboost.git
cd mlflow-xgboost
```

## Step 2: Set Up the Environment
Before running the project, make sure you have the necessary dependencies installed. You can create a virtual environment and install the required packages using pip:

```bash
pip install -r requirements.txt
```
requirements.txt should include necessary packages like mlflow, xgboost, sklearn, joblib, fastapi, and others.

## Step 3: Run the MLflow Project
Run the project using MLflow's mlflow run command. This will set up the environment, train the model, and log the experiment:

```bash
mlflow run .
```
This command will execute the project script and start the training of the XGBoost model with the specified hyperparameters.

## Step 4: Start the MLflow UI
Once the script runs successfully, you can start the MLflow UI to view your logged experiments. To do this, use the following command:

```bash
mlflow ui
```
By default, the MLflow UI will be accessible at http://localhost:5000.

## Step 5: View the Results
After starting the MLflow UI, open the following URL in your browser to view the logged experiment results:

```
http://localhost:5000
```
Here you will find the following information about the experiment:

### Parameters:
Hyperparameters: These include parameters such as learning_rate, max_depth, num_estimators, etc., that were used to train the XGBoost model.

### Metrics:
Accuracy: The accuracy of the trained model on the test set, which is logged as a metric in MLflow.

### Model:
Trained Model: The XGBoost model that was trained during the experiment. This model is logged in MLflow, allowing you to download and use it for future predictions.

## Step 6: Deployment of the Model
Once the model is trained and logged, you can deploy it as an API using FastAPI for serving the model predictions. Here’s an example for setting up a simple FastAPI app to serve the trained XGBoost model.

```
uvicorn deploy_model:app --reload
```

This will start the FastAPI server locally, and you can access it at http://127.0.0.1:8000. The /predict/ endpoint will allow you to send POST requests with the required features to get predictions from the XGBoost model.

## Example Request:
You can send a POST request with data to the /predict/ endpoint:
```
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
```
