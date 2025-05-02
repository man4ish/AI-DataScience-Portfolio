import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("titanic_data/train.csv")

# Select useful features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Drop rows with missing target
df = df[features + [target]].dropna(subset=[target])

# Fill missing values
imputer = SimpleImputer(strategy='most_frequent')

# Fill missing values
df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Embarked']]).ravel()
df['Age'] = SimpleImputer(strategy='median').fit_transform(df[['Age']]).ravel()
df['Fare'] = SimpleImputer(strategy='mean').fit_transform(df[['Fare']]).ravel()

# Encode categorical features
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Feature matrix and target
X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)
print("\nCross-validation Accuracy (5-fold):", np.round(cv_scores, 3))
print("Mean CV Accuracy:", round(cv_scores.mean(), 3))
