"""
House Price Prediction (Regression)

This script uses various regression models to predict housing prices from a Kaggle dataset.
It includes the following models:
- Ridge Regression
- Lasso Regression
- XGBoost Regression

The script performs the following tasks:
1. Loads and preprocesses the dataset.
2. Splits the data into training and testing sets.
3. Trains and evaluates the models using root mean squared error (RMSE).
4. Plots the feature importance for the XGBoost model.
5. Saves the trained models (Ridge, Lasso, XGBoost) to disk for later use.

Tech Stack:
- pandas: Data manipulation
- numpy: Array operations
- scikit-learn: Ridge, Lasso, and GridSearchCV for model training and evaluation
- xgboost: Gradient boosting model
- matplotlib: Plotting feature importance
- joblib: Saving and loading models

Author: Manish Kumar
Date: 2025-05-01
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the housing dataset.

    - Fills missing values with column means.
    - Converts categorical variables to numerical values using one-hot encoding.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Preprocessed feature set (X) and target variable (y).
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['SalePrice', 'Id'], axis=1)
    y = df['SalePrice']
    
    # Handle missing values by filling with column means
    X = X.fillna(X.mean())
    
    # Convert categorical variables to numerical using one-hot encoding
    X = pd.get_dummies(X)
    
    return X, y

def train_ridge_regression(X_train, y_train, X_test, y_test):
    """
    Train a Ridge regression model, perform hyperparameter tuning, and evaluate its performance.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target variable.

    Returns:
        float: RMSE of the trained Ridge regression model.
        model: The trained Ridge regression model.
    """
    ridge = Ridge()
    ridge_params = {'alpha': np.logspace(-6, 6, 13)}
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5)
    ridge_grid.fit(X_train, y_train)
    
    # Predict and evaluate the model
    ridge_pred = ridge_grid.best_estimator_.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    
    return ridge_rmse, ridge_grid.best_estimator_

def train_lasso_regression(X_train, y_train, X_test, y_test):
    """
    Train a Lasso regression model, perform hyperparameter tuning, and evaluate its performance.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target variable.

    Returns:
        float: RMSE of the trained Lasso regression model.
        model: The trained Lasso regression model.
    """
    lasso = Lasso()
    lasso_params = {'alpha': np.logspace(-6, 6, 13)}
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=5)
    lasso_grid.fit(X_train, y_train)
    
    # Predict and evaluate the model
    lasso_pred = lasso_grid.best_estimator_.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    
    return lasso_rmse, lasso_grid.best_estimator_

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model and evaluate its performance.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target variable.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): Test target variable.

    Returns:
        float: RMSE of the trained XGBoost model.
        model: The trained XGBoost model.
    """
    xgboost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    xgboost.fit(X_train, y_train)
    
    # Predict and evaluate the model
    xgboost_pred = xgboost.predict(X_test)
    xgboost_rmse = np.sqrt(mean_squared_error(y_test, xgboost_pred))
    
    return xgboost_rmse, xgboost

def plot_feature_importance(xgboost_model):
    """
    Plot the feature importance of the XGBoost model.

    Parameters:
        xgboost_model (xgb.XGBRegressor): The trained XGBoost model.
    """
    xgboost.plot_importance(xgboost_model)
    plt.show()

def main():
    """
    Main function to execute the model training, evaluation, and comparison of the regression models.

    Loads the data, trains three regression models (Ridge, Lasso, and XGBoost), 
    and evaluates their performance using RMSE. The models are then saved to disk.
    """
    # Load and preprocess the dataset
    X, y = load_and_preprocess_data('data/train.csv')
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate Ridge Regression
    ridge_rmse, ridge_model = train_ridge_regression(X_train, y_train, X_test, y_test)
    
    # Train and evaluate Lasso Regression
    lasso_rmse, lasso_model = train_lasso_regression(X_train, y_train, X_test, y_test)
    
    # Train and evaluate XGBoost
    xgboost_rmse, xgboost_model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Print the results
    print(f"Ridge Regression RMSE: {ridge_rmse:.3f}")
    print(f"Lasso Regression RMSE: {lasso_rmse:.3f}")
    print(f"XGBoost RMSE: {xgboost_rmse:.3f}")
    
    # Plot feature importance for XGBoost
    plot_feature_importance(xgboost_model)
    
    # Save the models to disk
    joblib.dump(ridge_model, 'ridge_model.pkl')
    joblib.dump(lasso_model, 'lasso_model.pkl')
    joblib.dump(xgboost_model, 'xgboost_model.pkl')
    print("Models saved successfully.")

if __name__ == '__main__':
    main()
