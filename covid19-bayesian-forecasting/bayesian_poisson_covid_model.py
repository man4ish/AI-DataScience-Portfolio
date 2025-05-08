"""
bayesian_poisson_covid_model.py

Bayesian Poisson Time-Series Forecasting of COVID-19 Cases Using PyMC

This script downloads real-world COVID-19 daily new case data from Our World In Data (OWID),
applies a Bayesian Poisson regression model using PyMC, and forecasts the next 14 days of
new case counts. The model includes uncertainty estimation (95% credible intervals).

Key Steps:
- Data fetching and preprocessing
- Model definition and sampling via PyMC
- Forecasting with uncertainty quantification
- Saving forecast plots to disk

Author: Manish Kumar
Date: 2025-07-05
"""

import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os

# ---------------------------
# Load and Prepare Real Data
# ---------------------------
def load_covid_data(country="India", days=100):
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url)
    df_country = df[df['location'] == country][['date', 'new_cases']].dropna()
    df_country = df_country.tail(days).reset_index(drop=True)
    df_country.to_csv("covid_daily_cases.csv", index=False)
    return df_country

# ---------------------------
# Bayesian Poisson Forecast
# ---------------------------
def bayesian_poisson_forecast(data, n_forecast=14):
    y_obs = data['new_cases'].values
    days = np.arange(len(y_obs))
    total_days = len(y_obs) + n_forecast

    with pm.Model() as model:
        # Priors for trend parameters
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=1)

        # Linear latent trend
        mu = pm.Deterministic("mu", alpha + beta * np.arange(total_days))

        # Poisson rate parameter
        lambda_ = pm.Deterministic("lambda", pm.math.exp(mu))

        # Observed data likelihood
        y_like = pm.Poisson("y_like", mu=lambda_[:len(y_obs)], observed=y_obs)

        # Forecasted future observations
        y_forecast = pm.Poisson("y_forecast", mu=lambda_[len(y_obs):], shape=n_forecast)

        # Sampling with increased tuning and draws
        trace = pm.sample(2000, tune=2000, target_accept=0.95, return_inferencedata=True)

    return trace, y_obs, days, n_forecast

# ---------------------------
# Plot & Save Results
# ---------------------------
def plot_forecast(trace, y_obs, days, n_forecast, output_path="covid19_forecast_bayesian.png"):
    forecast_samples = trace.posterior['y_forecast'].stack(samples=("chain", "draw")).values
    forecast_mean = forecast_samples.mean(axis=1)
    forecast_ci_lower = np.percentile(forecast_samples, 2.5, axis=1)
    forecast_ci_upper = np.percentile(forecast_samples, 97.5, axis=1)

    total_days = len(y_obs) + n_forecast

    plt.figure(figsize=(12, 5))
    plt.plot(days, y_obs, label="Observed")
    plt.plot(np.arange(len(y_obs), total_days), forecast_mean, label="Forecast", color='orange')
    plt.fill_between(np.arange(len(y_obs), total_days),
                     forecast_ci_lower, forecast_ci_upper,
                     color='orange', alpha=0.3, label="95% CI")
    plt.title("Bayesian Forecasting of COVID-19 Daily Cases")
    plt.xlabel("Days")
    plt.ylabel("Daily New Cases")
    plt.grid(True)
    plt.legend()

    plt.savefig(output_path, dpi=300)
    print(f"Forecast plot saved to: {output_path}")
    plt.close()

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    country_name = "India"
    forecast_days = 14

    print("Downloading COVID-19 data...")
    covid_data = load_covid_data(country=country_name)

    print("Running Bayesian forecast model...")
    trace, y_obs, days, n_forecast = bayesian_poisson_forecast(covid_data, forecast_days)

    # Plot the trace to inspect convergence
    az.plot_trace(trace)
    plt.savefig("trace_plot.png", dpi=300)  # Save the trace plot
    print("Trace plot saved to: trace_plot.png")

    print("Saving forecast plot...")
    plot_forecast(trace, y_obs, days, n_forecast)
