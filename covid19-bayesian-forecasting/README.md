# Bayesian Poisson COVID-19 Forecasting Model

This repository contains a Python script that applies Bayesian Poisson regression for forecasting COVID-19 cases. The model leverages PyMC (a powerful probabilistic programming framework) to estimate uncertainty in forecasts, allowing for more informed decision-making during disease spread analysis.

## Project Overview

The goal of this project is to build a Bayesian model to predict the future number of COVID-19 cases using historical data. By using a Poisson regression model, we can incorporate uncertainty into the forecast and generate credible intervals for the predicted number of cases.

### Key Features:
- Fetches COVID-19 daily case data for a specific country (e.g., India) from Our World in Data.
- Defines a Bayesian Poisson regression model using PyMC.
- Runs inference to estimate the posterior distribution of model parameters.
- Forecasts future cases and generates 95% credible intervals.
- Saves forecast and trace plots for analysis and diagnostics.

## Dependencies

To run this project, you need to install the following Python libraries. These can be installed using the provided `requirements.txt` file:

### Required Libraries:
- `pandas`: For data handling and preprocessing.
- `numpy`: For numerical operations.
- `matplotlib`: For plotting the results.
- `pymc`: For Bayesian modeling and sampling.
- `arviz`: For visualization and diagnostics of the Bayesian model.

### Installation

You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Usage
### Step 1: Download COVID-19 Data
The script fetches the COVID-19 data directly from the Our World in Data repository. The data will be loaded into a Pandas DataFrame and saved as covid_daily_cases.csv for further analysis.

### Step 2: Run Bayesian Poisson Forecasting
The script runs a Bayesian Poisson model to predict future COVID-19 cases. It uses PyMC to perform sampling via Markov Chain Monte Carlo (MCMC) methods. The model forecasts the next 14 days of new COVID-19 cases and generates the posterior distribution of predictions.

```bash
python bayesian_poisson_covid_model.py
```

## Output:
- covid_daily_cases.csv: A CSV file containing the historical daily cases for the selected country.

- covid19_forecast_bayesian.png: A forecast plot showing the observed and forecasted daily COVID-19 cases, with 95% credible intervals.

- trace_plot.png: A trace plot showing the convergence of the Markov Chains for the model parameters (used for diagnostics).

## Model Details
The model assumes that the number of daily new cases follows a Poisson distribution, with a log-linear mean (exponential link function). The model is defined as:

Poisson(lambda_t) where lambda_t = exp(alpha + beta * t) is the rate parameter (for time t).

alpha and beta are parameters that represent the intercept and trend of the model, respectively.

The sampling is performed using the NUTS (No-U-Turn Sampler) algorithm, and the forecast includes both the point predictions and the 95% credible intervals.

### Diagnostics and Convergence
The script uses arviz to plot the trace of the MCMC chains. If the chains have not fully converged, this will be evident in the trace plot, and adjustments to the model or sampling process can be made.