import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Step 1: Load the data (using Iris dataset for this example)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 1: Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Step 2: Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")

# Print Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 3: Implement k-fold cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")

# Step 4: Hyperparameter Tuning using GridSearchCV
# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"\nBest Hyperparameters from GridSearchCV: {grid_search.best_params_}")

# Evaluate on the test set with the best model
best_rf_classifier = grid_search.best_estimator_
y_pred_best = best_rf_classifier.predict(X_test)

# Evaluate the model
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"\nAccuracy with best model: {best_accuracy:.2f}")

# Print the Classification Report with the best model
print("\nClassification Report with best model:")
print(classification_report(y_test, y_pred_best))
