"""
Titanic Fare Prediction using Decision Tree Regressor

This script demonstrates the use of a Decision Tree Regressor to predict the ticket fare 
that Titanic passengers paid based on demographic and travel-related features. 
The model utilizes features like Age, Sex, Pclass, Embarked, FamilySize, and AgeGroup 
to predict the target variable "Fare". Additionally, family-related features and a 
logarithmic transformation of 'Fare' are used to improve model performance.

Steps performed in this script:
1. Load Titanic dataset.
2. Clean data (drop unnecessary columns, handle missing values).
3. Feature engineering (FamilySize, AgeGroup, additional family features, fare transformation).
4. Encode categorical variables.
5. Train a Decision Tree Regressor.
6. Tune hyperparameters using GridSearchCV.
7. Evaluate the model performance using Mean Squared Error (MSE) and R^2 Score.
8. Perform cross-validation to assess model stability.

Libraries used:
- pandas
- scikit-learn

Dataset:
- Titanic dataset from Kaggle (https://www.kaggle.com/c/titanic/data)

Usage:
Run the script to see model performance on the Titanic dataset.
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data cleaning
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)  # Drop unnecessary columns

# Replace missing values
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill missing Age values with the median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill missing Embarked values with the mode

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch']  # Create FamilySize feature
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 25, 35, 45, 55, 65, 100], 
                         labels=["0-12", "13-18", "19-25", "26-35", "36-45", "46-55", "56-65", "66+"])

# Additional family-related features
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)  # 1 if the passenger is alone, else 0
df['LargeFamily'] = (df['FamilySize'] > 3).astype(int)  # 1 if the family size is greater than 3

# Logarithmic transformation of 'Fare'
df['Fare'] = np.log1p(df['Fare'])  # Apply log(1 + Fare) to handle outliers and skewness

# Encoding categorical variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Encode 'Sex' as binary (0 = male, 1 = female)
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Encode 'Embarked' as categorical values
df['AgeGroup'] = label_encoder.fit_transform(df['AgeGroup'])  # Encode 'AgeGroup'

# Define feature variables and target
X = df[['Pclass', 'Sex', 'Age', 'FamilySize', 'Embarked', 'AgeGroup', 'IsAlone', 'LargeFamily']]
y = df['Fare']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=dt_regressor, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit grid search
grid_search.fit(X_train, y_train)

# Get best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best score (neg MSE):", grid_search.best_score_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Cross-validation performance
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"Cross-validation R^2 Scores: {cv_scores}")
print(f"Mean R^2 Score: {cv_scores.mean():.2f}")
