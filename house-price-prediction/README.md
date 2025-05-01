# House Price Prediction (Regression)

## Overview
This project demonstrates the process of predictive modeling using Ridge, Lasso, and XGBoost on a Kaggle housing dataset. The goal is to predict house prices based on various features like the number of bedrooms, square footage, year built, and more. This repository showcases the use of regularization techniques, hyperparameter tuning, and feature selection in a regression context.

## Tech Stack
- **Python**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and utilities
- **XGBoost** - Gradient boosting framework
- **matplotlib** - Plotting and data visualization

## Focus
- Feature selection
- Regularization (Ridge and Lasso)
- Hyperparameter tuning
- Model evaluation

## Project Structure
```bash
.
├── data
│   ├── train.csv       # Training data
│   └── test.csv        # Test data
├── notebooks
│   └── house_price_prediction.ipynb # Jupyter notebook with the main analysis
├── train_model.py      # Script to train the model
├── requirements.txt    # List of dependencies
└── README.md           # Project documentation
```

# Steps to Run the Project
## Step 1: Clone the Repository
Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

## Step 2: Install Dependencies
Make sure you have pip installed, then run:

```
bash
pip install -r requirements.txt
```

## Step 3: Run the Model Training Script
Once the dependencies are installed, you can train the model using the following command:

```
bash
python train_model.py
```

This will:

- Load the data from data/train.csv.

- Preprocess the data.

- Train the model using Ridge, Lasso, and XGBoost regression techniques.

- Save the trained models and evaluation results.

## Step 4: View the Results
The script outputs the performance of each model and the trained model can be used to make predictions on new data.

### Example train.csv
Here's an example of what the train.csv data might look like:

```
Id,OverallQual,GrLivArea,GarageCars,TotRmsAbvGrd,YearBuilt,SalePrice
1,7,1710,2,8,2003,208500
2,6,1262,2,6,1976,181500
3,7,1786,3,7,2001,223500
4,8,2392,3,9,2000,250000
5,5,1158,1,6,1993,140000
6,6,1310,2,6,2004,250000
7,7,1800,2,8,1997,175000
8,5,1630,2,7,1995,160000
9,8,1998,3,9,2007,400000
10,6,1254,1,6,1998,155000
```

## Columns:
- Id: A unique identifier for each house.

- OverallQual: Overall quality of the house (scale of 1-10).

- GrLivArea: Ground living area (square feet).

- GarageCars: Number of cars that can fit in the garage.

- TotRmsAbvGrd: Total rooms above grade.

- YearBuilt: Year the house was built.

- SalePrice: The target variable (price the house was sold for).

## Model Evaluation
The evaluation is based on the Root Mean Squared Error (RMSE). The lower the RMSE, the better the model's performance.

## Requirements
```
pandas==1.5.3
scikit-learn==1.1.2
xgboost==1.7.5
matplotlib==3.7.1
joblib==1.1.0
```

## Conclusion
This project demonstrates the application of regression models such as Ridge, Lasso, and XGBoost to predict house prices. Regularization techniques like Ridge and Lasso help to prevent overfitting, and XGBoost provides a powerful gradient boosting solution for regression tasks. The models can be further improved with hyperparameter tuning and additional feature engineering.