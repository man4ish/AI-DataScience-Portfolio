"""
This script uses a Naive Bayes classifier to predict the presence of diabetes based on the Pima Indians Diabetes Dataset. 
The model predicts whether a patient has diabetes or not based on medical diagnostic features such as glucose levels, blood pressure, BMI, etc.

Steps:
1. Load the dataset from a URL and assign column names.
2. Handle missing data by replacing zeros with NaN and imputing the missing values with the mean.
3. Split the dataset into features (X) and target (y), where the target is the 'Outcome' column (1 for diabetic, 0 for non-diabetic).
4. Split the data into training and testing sets (80% train, 20% test).
5. Scale the features using StandardScaler to normalize the values for better Naive Bayes performance.
6. Apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
7. Initialize and train the Naive Bayes model using the Gaussian Naive Bayes method.
8. Perform k-fold cross-validation to check for overfitting.
9. Evaluate the model using accuracy, precision, recall, F1-score, and classification report.

Modules used:
- pandas: For data manipulation and handling.
- sklearn.model_selection: For splitting the dataset into training and testing sets and performing cross-validation.
- sklearn.preprocessing: For scaling features using StandardScaler.
- imbalanced-learn: For handling class imbalance with SMOTE.
- sklearn.naive_bayes: For implementing the Gaussian Naive Bayes classifier.
- sklearn.metrics: For evaluating model performance (classification report, precision, recall, etc.).

"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Handle missing values (replace zeros with NaN and then impute)
df.replace(0, pd.NA, inplace=True)
df.fillna(df.mean(), inplace=True)

# Split into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check if both classes are present in y_train
print(f"Class distribution in y_train: {y_train.value_counts()}")

# Only apply SMOTE if both classes are present
if len(y_train.value_counts()) > 1:
    # Handle class imbalance using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
else:
    print("Warning: Target variable contains only one class in the training set. SMOTE cannot be applied.")

# Feature scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled) if len(y_train.value_counts()) > 1 else scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Naive Bayes model
nb = GaussianNB()

# Train the model with resampled data if SMOTE was applied
if len(y_train.value_counts()) > 1:
    nb.fit(X_resampled, y_resampled)
else:
    nb.fit(X_train, y_train)

# Cross-validation to check for overfitting
cv_scores = cross_val_score(nb, X_resampled if len(y_train.value_counts()) > 1 else X_train, 
                            y_resampled if len(y_train.value_counts()) > 1 else y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {cv_scores.mean()}")

# Evaluate the model on the test set
accuracy = nb.score(X_test, y_test)
print(f"Test accuracy: {accuracy}")

# Make predictions on the test set
y_pred = nb.predict(X_test)

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Additional performance metrics: Precision, Recall, F1-Score
print(f"Precision: {accuracy_score(y_test, y_pred)}")

