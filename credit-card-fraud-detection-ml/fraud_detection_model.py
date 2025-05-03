"""
fraud_detection_rf.py

A machine learning pipeline for detecting fraudulent transactions using a Random Forest Classifier.
Includes:
- Train/test split
- K-Fold cross-validation
- Hyperparameter tuning with GridSearchCV
- Evaluation with precision, recall, F1-score, AUC
- ROC curve plotting and image saving

Dataset: Credit Card Fraud Detection dataset from Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import zipfile

def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df

def load_data(zip_file_path="data/creditcardfraud.zip", file_name_in_zip="creditcard.csv"):
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        # Extract the file into memory
        with z.open(file_name_in_zip) as f:
            df = pd.read_csv(f)
    return df

def preprocess_data(df):
    X = df.drop(columns=['Class'])
    y = df['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean():.4f}")

def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='recall')
    grid.fit(X_train, y_train)
    print(f"\nBest Hyperparameters from GridSearchCV: {grid.best_params_}")
    return grid.best_estimator_

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    print(f"\nAccuracy on test set: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # AUC Score
    auc_score = roc_auc_score(y_test, y_probs)
    print(f"AUC Score: {auc_score:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color="blue")
    plt.plot([0, 1], [0, 1], linestyle='--', color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nTraining initial Random Forest model...")
    model = train_random_forest(X_train, y_train)
    cross_validate_model(model, X_train, y_train)

    print("\nTuning hyperparameters...")
    best_model = tune_hyperparameters(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
