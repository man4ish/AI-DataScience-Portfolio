"""
This module implements a Logistic Regression model for predicting survival on the Titanic dataset.

Steps involved in this module:
1. Loads the Titanic dataset from a provided URL.
2. Preprocesses the data by handling missing values and performing one-hot encoding on categorical features.
3. Splits the dataset into training and test sets.
4. Scales the features using StandardScaler for normalization.
5. Trains a Logistic Regression model on the training data.
6. Makes predictions on the test data.
7. Evaluates the model using accuracy, confusion matrix, and classification report.
8. Saves the confusion matrix plot as an image file (confusion_matrix.png).
9. Displays the confusion matrix plot.

The Titanic dataset can be found at: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Dependencies:
- pandas
- numpy
- sklearn
- seaborn
- matplotlib

"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def logistic_regression_titanic():
    """
    This function implements Logistic Regression to predict survival on the Titanic dataset.
    It performs the following steps:
    1. Loads the Titanic dataset.
    2. Preprocesses the data (handling missing values, one-hot encoding categorical features).
    3. Splits the data into training and testing sets.
    4. Normalizes the feature values using StandardScaler.
    5. Trains a Logistic Regression model.
    6. Evaluates the model using accuracy, confusion matrix, and classification report.
    7. Saves the confusion matrix as an image.

    Returns:
        None
    """

    # Load Titanic dataset (assuming the dataset is in the current directory)
    # You can download the dataset from Kaggle and place it in the working directory as "titanic.csv"
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)

    # Preview the dataset
    print(df.head())

    # Data Preprocessing
    # Handling missing values
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)  # Dropping non-numerical columns

    # Convert categorical columns into numeric using one-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Feature selection
    X = df.drop(columns=['Survived'])  # Features
    y = df['Survived']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling (normalizing the data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot as an image
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Call the function to execute
logistic_regression_titanic()
