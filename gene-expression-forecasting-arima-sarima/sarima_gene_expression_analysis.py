import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# --- Step 1: Load Gene Expression Data ---
# For this example, let's generate some sample data (Gene expression levels during the cell cycle).
# Replace this with real data (e.g., load a CSV file) when you have it.

data = {
    'Time (hrs)': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114],
    'Gene Expression Level': [10, 15, 20, 25, 30, 27, 22, 18, 12, 14, 17, 19, 22, 24, 26, 28, 30, 31, 32, 33]
}



# Convert to DataFrame
df = pd.DataFrame(data)

df = pd.DataFrame(data)

# Apply diff()
df['Diff'] = df['Gene Expression Level'].diff().dropna()

df.set_index('Time (hrs)', inplace=True)

# --- Step 2: Plot the Data ---
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Gene Expression Level'], marker='o', color='b')
plt.title('Gene Expression Levels Over Time (Cell Cycle)')
plt.xlabel('Time (hrs)')
plt.ylabel('Gene Expression Level')
plt.grid(True)

# Save the plot image
plt.savefig('gene_expression_levels.png')

# Show the plot
plt.show()

# --- Step 3: Check for Stationarity (ADF Test) ---
# Perform Augmented Dickey-Fuller test to check for stationarity
result = adfuller(df['Gene Expression Level'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] <= 0.05:
    print("The data is stationary.")
else:
    print("The data is not stationary. We need to difference it.")

# --- Step 4: Fit the SARIMA Model ---
# Since the data is non-stationary (based on the ADF test result), we will apply differencing.
# We'll use SARIMA with seasonal components (m=24, assuming daily cycle).

# Fit SARIMA model (ARIMA order=(1,1,1), Seasonal order=(1,1,1,24))
sarima_model = SARIMAX(df['Gene Expression Level'],
                       order=(1, 1, 1),  # ARIMA orders (p, d, q)
                       seasonal_order=(1, 1, 1, 24),  # Seasonal ARIMA orders (P, D, Q, m)
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Fit the model
sarima_result = sarima_model.fit()

# Print the summary of the model
print(sarima_result.summary())

# --- Step 5: Forecast Future Gene Expression Levels ---
# Forecast next 12 time points (e.g., next 72 hours with 6-hour intervals)

forecast_steps = 12
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_index = np.arange(df.index[-1] + 6, df.index[-1] + 6 * (forecast_steps + 1), 6)  # 6-hour intervals

# --- Step 6: Plot Observed vs Forecasted Gene Expression Levels ---
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Gene Expression Level'], marker='o', label='Observed', color='blue')
plt.plot(forecast_index, forecast.predicted_mean, color='red', marker='x', label='Forecasted')
plt.title('Gene Expression Forecast (Cell Cycle)')
plt.xlabel('Time (hrs)')
plt.ylabel('Gene Expression Level')
plt.legend()
plt.grid(True)

# Save the forecasted plot image
plt.savefig('gene_expression_forecast.png')

# Show the forecasted plot
plt.show()

# --- Step 7: Save the Forecast Results ---
forecast_df = pd.DataFrame({
    'Time (hrs)': forecast_index,
    'Forecasted Gene Expression Level': forecast.predicted_mean
})

# Save forecasted results to CSV (if needed)
forecast_df.to_csv('forecasted_gene_expression.csv', index=False)
print("Forecast results saved to 'forecasted_gene_expression.csv'")

