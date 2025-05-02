"""
Breast Cancer Classification using Support Vector Machine (SVM)

This script performs a classification task to predict whether breast cancer tumors are malignant or benign
using the Support Vector Machine (SVM) algorithm. It utilizes the popular UCI Breast Cancer dataset
for model training and evaluation.

Steps followed:
1. Load and preprocess the breast cancer dataset.
2. Split the data into training and testing sets.
3. Train an SVM classifier with a linear kernel.
4. Evaluate the model's performance using accuracy and a confusion matrix.
5. Visualize the decision boundary for better understanding.

Dependencies:
- numpy
- pandas
- scikit-learn
- matplotlib

Usage:
1. Install the required libraries using `pip install -r requirements.txt` (if provided).
2. Run the script to train and test the model.
3. The script will output the model's accuracy, confusion matrix, and decision boundary plot.

Author:
Manish Kumar

"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM classifier with RBF kernel
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_clf.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=data.target_names))

