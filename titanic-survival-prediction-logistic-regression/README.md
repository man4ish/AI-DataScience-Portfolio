# Titanic Survival Prediction with Logistic Regression

This repository contains a machine learning project that predicts the survival of passengers on the Titanic using the Logistic Regression algorithm. The dataset is sourced from Kaggle's Titanic competition, which is a well-known dataset for binary classification tasks.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Project Description

This project aims to predict whether a Titanic passenger survived or not based on a set of features, including age, sex, class, and more. The algorithm used to make the predictions is Logistic Regression, a commonly used method for binary classification problems.

## Dataset

The dataset used for this project is the **Titanic: Machine Learning from Disaster** dataset, available on Kaggle. It contains the following columns:

- `PassengerId`: A unique identifier for each passenger.
- `Pclass`: The passenger's class (1 = First, 2 = Second, 3 = Third).
- `Name`: The passenger's name.
- `Sex`: The passenger's sex (male or female).
- `Age`: The passenger's age in years.
- `SibSp`: The number of siblings or spouses the passenger had aboard.
- `Parch`: The number of parents or children the passenger had aboard.
- `Ticket`: The ticket number.
- `Fare`: The fare the passenger paid.
- `Cabin`: The cabin where the passenger stayed.
- `Embarked`: The port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).
- `Survived`: The target variable (0 = No, 1 = Yes), indicating whether the passenger survived.

## Installation

To run this project, you need to have Python installed along with some dependencies. You can install them using `pip`.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-logistic-regression.git
   cd titanic-logistic-regression
   ```

Install the required packages:

```
bash
pip install -r requirements.txt
```

### Usage
Download the Titanic dataset from Kaggle and save it as train.csv in the project directory.

### Run the Logistic Regression model:

```
bash
python logistic_regression_titanic.py
```

This script will load the Titanic dataset, preprocess the data, train the Logistic Regression model, and output the accuracy and other evaluation metrics.

### Results
The model will output the classification accuracy and the confusion matrix, showing how well the model has predicted the survival of passengers.

### Example output:
```
Accuracy: 0.81
Confusion Matrix:
[[503  13]
 [ 99 149]]
``` 
