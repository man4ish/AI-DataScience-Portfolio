"""
Customer Churn Prediction

This script builds a supervised machine learning model to predict customer churn
using logistic regression and random forests. It handles data preprocessing, 
feature engineering, class imbalance handling, model training, and evaluation.

Tech Stack:
- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn (for SMOTE)

Steps:
1. Load and preprocess the dataset.
2. Handle missing values and categorical variables.
3. Split the data into training and testing sets.
4. Apply SMOTE to handle class imbalance.
5. Train Logistic Regression and Random Forest models.
6. Evaluate the models with classification reports, confusion matrices, and AUC-ROC scores.
7. Visualize the feature importance from the Random Forest model.
8. Generate performance metrics and visualizations.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset (Replace with your actual dataset)
data = pd.read_csv('customer_churn_data.csv')

# Display first few rows of the dataset
print(data.head())

# Step 1: Data Preprocessing
# Handling missing values (if any)
data = data.fillna(data.mean())  # Replace missing numerical values with column mean

# Convert categorical variables to numerical (One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)

# Step 2: Feature Engineering
# Separate features and target variable
X = data.drop('churn', axis=1)  # 'churn' is the target variable
y = data['churn']

# Step 3: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Handle Class Imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 6: Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train_resampled, y_train_resampled)

# Step 7: Random Forest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Step 8: Model Evaluation
# Logistic Regression Evaluation
y_pred_logreg = logreg.predict(X_test)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]))

# Random Forest Evaluation
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Random Forest AUC-ROC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# Step 9: Visualize Confusion Matrix
# Plot confusion matrix for Random Forest
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Feature Importance from Random Forest
feature_importance = rf.feature_importances_
features = X.columns

# Visualize the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()
