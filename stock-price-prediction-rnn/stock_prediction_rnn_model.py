"""
Stock Price Prediction using LSTM (Long Short-Term Memory) Neural Network

This script demonstrates the use of an LSTM model to predict the stock prices of a given company.
The model is trained on historical stock price data and aims to predict future prices based on patterns 
in the past data. This script includes steps for data preprocessing, feature engineering, model building, 
and training the model.

Steps performed in this script:
1. Download stock data using Yahoo Finance (yfinance library).
2. Preprocess the data (normalize and prepare the data for LSTM model).
3. Define LSTM model architecture.
4. Train the LSTM model.
5. Evaluate the model performance using Mean Squared Error (MSE).
6. Plot the predicted vs actual values.

Libraries used:
- pandas
- numpy
- yfinance
- keras
- tensorflow

Usage:
1. Install the required libraries: `pip install yfinance tensorflow numpy pandas keras matplotlib`
2. Run the script to see the stock price prediction for the given stock (AAPL in this case).
"""

import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

# Disable GPU if not using for computation (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces TensorFlow to use CPU

def download_data(stock_ticker, start_date, end_date):
    """
    Downloads historical stock data using the Yahoo Finance API.
    
    Parameters:
    stock_ticker (str): Stock symbol for the company (e.g., 'AAPL').
    start_date (str): Start date for data download in 'YYYY-MM-DD' format.
    end_date (str): End date for data download in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: A pandas DataFrame containing the historical stock data.
    """
    return yf.download(stock_ticker, start=start_date, end=end_date)

def preprocess_data(stock_data, time_steps=60):
    """
    Preprocess the stock data to prepare it for LSTM model training.
    
    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock prices.
    time_steps (int): The number of previous days' data used to predict the next day's price.
    
    Returns:
    np.ndarray: Processed feature and target arrays ready for training the model.
    """
    # Use only 'Close' prices for prediction
    stock_data = stock_data[['Close']]
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)
    
    # Prepare features (X) and target (y)
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])  # Last 'time_steps' days as features
        y.append(scaled_data[i, 0])  # Next day's close price as target
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be compatible with LSTM (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def build_model(input_shape):
    """
    Builds and compiles the LSTM model for stock price prediction.
    
    Parameters:
    input_shape (tuple): The shape of the input data (time_steps, features).
    
    Returns:
    keras.models.Sequential: A compiled LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # Fix the warning with Input layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def plot_results(test_data, predicted_data, filename="stock_price_prediction.png"):
    """
    Plots the actual vs predicted stock prices for evaluation and saves the plot as an image.
    
    Parameters:
    test_data (pd.DataFrame): Actual stock prices from the test set.
    predicted_data (np.ndarray): Predicted stock prices by the model.
    filename (str): The name of the file to save the plot as. Default is "stock_price_prediction.png".
    """
    plt.plot(test_data, color='blue', label='Actual Stock Price')
    plt.plot(predicted_data, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()

    # Save the plot as a PNG image
    plt.savefig(filename)

    # Optionally, display the plot as well
    plt.show()

def main():
    """
    Main function to execute the stock prediction pipeline.
    It downloads data, preprocesses it, builds and trains the model, and visualizes the predictions.
    """
    # Step 1: Download data
    stock_data = download_data('AAPL', '2010-01-01', '2022-01-01')
    
    # Step 2: Preprocess data
    time_steps = 60
    X, y, scaler = preprocess_data(stock_data, time_steps)
    
    # Step 3: Split data into training and testing sets (80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Step 4: Build and train the model
    model = build_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Step 5: Predict on the test data
    y_pred = model.predict(X_test)
    
    # Reverse the scaling for predicted values
    y_pred = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform([y_test])
    
    # Step 6: Plot the results
    plot_results(y_test_actual[0], y_pred)

if __name__ == "__main__":
    main()
