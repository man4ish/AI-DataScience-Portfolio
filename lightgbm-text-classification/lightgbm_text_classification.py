"""
LightGBM Binary Classification Example

This script demonstrates how to use the LightGBM (Light Gradient Boosting Machine) library for a binary classification task.
It uses the Adult Income dataset to classify whether an individual's income is greater than 50K (binary classification: 0 or 1).

Steps followed in the script:
1. Load the dataset from a given URL and handle missing values.
2. Convert categorical features into numeric format using one-hot encoding.
3. Split the dataset into features (X) and target (y).
4. Split the data into training and testing sets.
5. Define the hyperparameters for the LightGBM model.
6. Train the model using the training dataset.
7. Make predictions on the test dataset.
8. Evaluate the model's performance using accuracy as the metric.

Modules used:
- `pandas`: Data manipulation and preprocessing.
- `scikit-learn`: Model evaluation, train-test splitting, and accuracy calculation.
- `lightgbm`: Training a gradient boosting model for binary classification.

Input Data:
- The dataset used in this script is the "Adult Income" dataset, available at: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- The dataset includes demographic information and income level (whether it is above 50K or not).

Output:
- The script prints the accuracy of the trained model on the test data.

Note:
- Ensure the dataset URL or path is accessible, and the correct columns are used.
- Adjust hyperparameters and model parameters based on specific use cases for better performance.

Example:
    - Run the script as a standalone program to train and evaluate a LightGBM model on the Adult Income dataset.

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Load dataset (replace with your own dataset path)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

data = pd.read_csv(url, names=columns, sep=',\s', engine='python')

# Handle missing values
data = data.replace(' ?', pd.NA).dropna()

# Convert categorical columns to numeric using one-hot encoding
categorical_columns = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
data = pd.get_dummies(data, columns=categorical_columns)

# Split into features (X) and target (y)
X = data.drop('income', axis=1)  # Drop target column
y = (data['income'] == ' >50K').astype(int)  # Convert target to binary (0 or 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set hyperparameters for the LightGBM model
params = {
    'objective': 'binary',       # Binary classification
    'metric': 'binary_error',    # Metric for binary classification (error rate)
    'boosting_type': 'gbdt',     # Gradient Boosting Decision Tree
    'num_leaves': 31,            # Number of leaves in each tree
    'learning_rate': 0.05,       # Learning rate
    'feature_fraction': 0.9      # Fraction of features to use in each iteration
}

# Train the model
num_round = 100  # Number of boosting rounds
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)

# Make predictions
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

# Convert probabilities to binary values (0 or 1)
y_pred_binary = (y_pred >= 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.4f}')
