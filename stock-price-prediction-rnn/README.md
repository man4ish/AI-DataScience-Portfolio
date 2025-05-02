# Stock Price Prediction using LSTM (Long Short-Term Memory)

This repository contains a script for predicting stock prices using an LSTM (Long Short-Term Memory) neural network. The model is trained on historical stock price data and aims to predict future stock prices based on past patterns.

## Overview

This project demonstrates how to use LSTM to predict the stock price of a given company. The model is trained on historical data, which is fetched using the Yahoo Finance API (`yfinance`), and predictions are visualized to compare with the actual stock prices.

### Steps performed in this project:
1. Download historical stock data from Yahoo Finance.
2. Preprocess and normalize the data to make it suitable for LSTM.
3. Define and build an LSTM neural network model.
4. Train the model using the preprocessed data.
5. Evaluate the model performance using Mean Squared Error (MSE) and visualizations.
6. Plot the predicted vs actual stock prices.

## Libraries Required

- `yfinance`: To download stock data.
- `tensorflow`: To build and train the LSTM model.
- `keras`: To define the LSTM layers.
- `numpy`: For numerical operations.
- `pandas`: For data manipulation.
- `matplotlib`: For visualizing results.

To install these libraries, run:

```bash
pip install yfinance tensorflow numpy pandas keras matplotlib
```

## Run

```bash
python stock_prediction_rnn_model.py
```
This will run the script, download data for Apple (AAPL) from 2010 to 2022, preprocess it, train the LSTM model, and visualize the results.

## Results
The model will plot a graph showing the actual stock prices in blue and the predicted stock prices in red. The comparison allows you to see how well the model has learned the underlying trends in the stock data.

## Model Performance
The performance of the model can be evaluated using metrics such as Mean Squared Error (MSE) and R² Score, which are used to quantify how well the model is predicting future stock prices.

## Future Improvements
Hyperparameter tuning: Fine-tuning the model architecture and training process can improve the accuracy of predictions.

- Incorporating other features: Additional features such as trading volume, technical indicators (e.g., RSI, moving averages), or news sentiment could be added to improve the model’s performance.

- Use of more advanced models: Exploring advanced models like GRU (Gated Recurrent Unit) or Transformer-based architectures for time-series forecasting.