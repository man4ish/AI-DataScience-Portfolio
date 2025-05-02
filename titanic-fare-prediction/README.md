# Titanic Fare Prediction using Decision Tree Regressor

This script demonstrates the use of a Decision Tree Regressor to predict the ticket fare that Titanic passengers paid based on demographic and travel-related features. The model utilizes features like Age, Sex, Pclass, Embarked, FamilySize, AgeGroup, and new features such as IsAlone and LargeFamily to predict the target variable "Fare".

## Steps performed in this script:
1. **Load Titanic dataset**: The dataset is loaded from a public URL.
2. **Data Cleaning**: 
   - Drop unnecessary columns (`Name`, `Ticket`, `Cabin`).
   - Handle missing values in `Age` and `Embarked`.
3. **Feature Engineering**:
   - Create new features like `FamilySize` (sum of `SibSp` and `Parch`) and `AgeGroup` (categorical age ranges).
   - Create new features such as `IsAlone` (a boolean indicating whether a passenger is traveling alone) and `LargeFamily` (family size greater than 3).
   - Apply a logarithmic transformation on the `Fare` feature to handle outliers.
4. **Encode Categorical Variables**: The categorical features like `Sex`, `Embarked`, and `AgeGroup` are encoded using Label Encoding.
5. **Train a Decision Tree Regressor**: A Decision Tree Regressor model is trained on the processed data.
6. **Hyperparameter Tuning using GridSearchCV**: The model's hyperparameters are optimized using GridSearchCV to find the best configuration.
7. **Evaluate Model Performance**:
   - The model performance is evaluated using **Mean Squared Error (MSE)** and **R² Score**.
   - **Cross-validation** is performed to assess model stability and generalization.
8. **Results**: The best hyperparameters, MSE, R² score, and cross-validation results are displayed.

## Libraries used:
- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For building and evaluating the Decision Tree Regressor model.
- **numpy**: For numerical operations.
- **matplotlib (optional)**: For visualizing feature importance (if included in the script).

## Dataset:
- **Titanic dataset** from Kaggle: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

## Usage:
1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/data-science-portfolio.git
   ```

## Run the script:

```bash
python titanic_decision_tree_regressor.py
```

## View Results:
The script will print the best hyperparameters, mean squared error (MSE), R² score, and the cross-validation results to the console.

## Example output:

```bash
Best parameters: {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 2}
Best score (neg MSE): -0.29888954850707006
Mean Squared Error: 0.27
R^2 Score: 0.68
Cross-validation R^2 Scores: [0.73579725 0.75005094 0.70946506 0.55940056 0.63610044]
Mean R^2 Score: 0.68
```

## Future Work:
- Experiment with other models (e.g., Random Forest, Gradient Boosting) for potentially better performance.
- Further feature engineering and data transformations could be explored to improve accuracy.