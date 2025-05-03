"""
random_forest_regression_example.py

This script demonstrates how to use a Random Forest Regressor on a simple real-world-like dataset.
The dataset consists of (X, Y) pairs where X is the input feature and Y is the target value.

Steps:
1. Load the dataset from a CSV file.
2. Train a Random Forest regression model.
3. Predict target values over a test range.
4. Visualize and save the prediction plot.

Author: Manish Kumar
Date: 2015-05-02
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("data/random-forest-regression-dataset.csv", sep=';', header=None, names=["X", "Y"])

# Prepare input and output
X = df[["X"]]  # input feature as 2D
y = df["Y"]    # target

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on test range
X_test = np.arange(0, 12, 0.1).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="red", label="Actual")
plt.plot(X_test, y_pred, color="blue", label="Prediction")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random Forest Regression")
plt.legend()
plt.grid(True)

# Save figure
plt.savefig("random_forest_regression_plot.png")
plt.show()
