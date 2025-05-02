# Titanic Survival Prediction using Decision Tree

This project uses the full Titanic dataset to predict passenger survival using a Decision Tree Classifier. It includes preprocessing, encoding, and cross-validation.

## Dataset

Download from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data) and place `train.csv` in this directory.

## Features Used

- `Pclass` (Passenger Class)
- `Sex`
- `Age`
- `SibSp` (Siblings/Spouses aboard)
- `Parch` (Parents/Children aboard)
- `Fare`
- `Embarked` (Port of embarkation)

## How to Run

1. Clone this repo
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn numpy
   ```
3. Run the script:
   ```bash
   python titanic_decision_tree.py  
   ``` 

## Output
- Accuracy and classification report

- 5-fold cross-validation accuracy