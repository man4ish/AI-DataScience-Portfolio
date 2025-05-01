# Customer Churn Prediction

## Overview

This project aims to predict customer churn (subscription cancellations) using supervised machine learning techniques. The model is built using **Logistic Regression** and **Random Forests**, with a focus on feature engineering, handling class imbalance, and model interpretation.

### Objective:
Predict customer churn based on various customer data features to help businesses retain valuable customers.

## Tech Stack

- **Python**: Primary programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning models and metrics
- **imbalanced-learn**: Techniques to handle imbalanced datasets

## Installation

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

## Step 2: Install Required Dependencies
Use the following command to install all required dependencies:

```bash
pip install -r requirements.txt
```

## Step 3: Run the Model
You can train and test the model by running the following script:

```bash
python train_model.py
```
This script performs the following actions:

- Loads the dataset.

- Preprocesses the data (handling missing values, encoding categorical variables, etc.).

- Splits the dataset into training and testing sets.

- Trains a logistic regression model and a random forest classifier.

- Evaluates the models and logs results.

- Prints the accuracy of the models.

## Step 4: Evaluate the Model
The results will be logged in the console. You will see the accuracy of the model on the test dataset printed out.

Project Structure
```
customer-churn-prediction/
├── data/
│   └── churn_data.csv        # Sample customer churn data
├── train_model.py            # Script to train and evaluate the churn prediction model
├── requirements.txt          # Python dependencies for the project
├── README.md                 # Project documentation

```

## Dataset
The dataset used in this project contains various customer attributes, such as:

- CustomerID: Unique ID of the customer

- Gender: Gender of the customer

- Age: Age of the customer

- MonthlySpend: Monthly spend of the customer

- ContractType: Type of subscription contract

- IsChurned: Target variable (1 if the customer has churned, 0 otherwise)

## Sample Data (churn_data.csv):
```
CustomerID,Gender,Age,MonthlySpend,ContractType,IsChurned
1,Male,30,50,Month-to-Month,0
2,Female,45,60,One-Year,1
3,Female,34,50,Month-to-Month,0
...

```

## Model Interpretation
The project evaluates the accuracy of the models (Logistic Regression and Random Forest) using the test data. The output is displayed in the console as:

```
Model Accuracy (Logistic Regression): 0.85
Model Accuracy (Random Forest): 0.88
Handling Class Imbalance
```

In this project, we also use techniques from imbalanced-learn to handle class imbalance if the target variable (IsChurned) is highly imbalanced. We perform oversampling of the minority class using SMOTE (Synthetic Minority Over-sampling Technique).

## Conclusion
This project demonstrates how machine learning techniques can be applied to predict customer churn, helping businesses retain customers. By leveraging models like Logistic Regression and Random Forests, and addressing class imbalance issues, we can build an effective prediction system.