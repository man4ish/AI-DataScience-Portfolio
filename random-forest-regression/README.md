# Random Forest Regression Example

This repository demonstrates how to use a Random Forest Regressor from `scikit-learn` on a simple, curated dataset for educational and practice purposes.

## Overview

Random Forest is an ensemble learning method that performs regression by averaging the output of multiple decision trees. This project walks through:

- Loading a dataset
- Training a Random Forest Regressor
- Visualizing predictions
- Saving the results

## Dataset

**Filename**: `random-forest-regression-dataset.csv`  
**Format**: CSV (semicolon-separated)

### Sample Data

```
X;Y
1;100
2;80
3;70
4;60
5;50
6;40
7;30
8;20
9;10
10;5
```

- **X**: A single numerical feature (e.g., time, units, or arbitrary numeric input)
- **Y**: Target values decreasing with X (a non-linear trend)

This dataset is synthetically created to resemble a downward trend and is ideal for regression practice.  

You can expand or replace it with real-world datasets (e.g., from Kaggle).

## How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib
```
### 2. Prepare dataset
Ensure random-forest-regression-dataset.csv is in the project folder.

### 3. Run the script
```bash
python random_forest_regression_example.py
```

## Output: 
A plot will be saved as random_forest_regression_plot.png.

Actual vs. predicted regression will be visualized.