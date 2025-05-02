# Diabetes Prediction Using Naive Bayes

This Python script uses the Naive Bayes classifier to predict the presence of diabetes based on the Pima Indians Diabetes Dataset. The model predicts whether a patient has diabetes (Outcome = 1) or not (Outcome = 0) based on medical diagnostic features such as glucose levels, blood pressure, BMI, and more.

## Steps in the Script

### Data Loading:
- The dataset is loaded from a public URL (https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) and column names are assigned to the dataset.

### Handling Missing Data:
- Zeros in the dataset are replaced with NaN, and missing values are imputed with the column mean using `fillna()`.

### Feature-Target Split:
- The data is split into features (X) and the target variable (y). The target is the `Outcome` column, where 1 represents a diabetic patient and 0 represents a non-diabetic patient.

### Train-Test Split:
- The data is split into training and testing sets using an 80/20 ratio, and the stratified split ensures the target variable's distribution is preserved.

### Class Imbalance Handling:
- **SMOTE** (Synthetic Minority Over-sampling Technique) is used to handle class imbalance by generating synthetic samples for the minority class in the training data.

### Feature Scaling:
- The features are scaled using **StandardScaler** to normalize the values, which improves the performance of the Naive Bayes classifier.

### Naive Bayes Model:
- A **Gaussian Naive Bayes** model is initialized and trained on the resampled training data (if SMOTE is applied).

### Cross-Validation:
- The model undergoes **5-fold cross-validation** to check for overfitting and ensure the model generalizes well.

### Model Evaluation:
- The model's performance is evaluated on the test set using accuracy, precision, recall, F1-score, and a classification report.

## Requirements

The following Python libraries are required to run this script:

- `pandas`: For data manipulation and handling.
- `scikit-learn`: For machine learning functionalities, including train-test split, Naive Bayes model, scaling, and cross-validation.
- `imblearn`: For handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique).
- `numpy`: For numerical operations.

You can install the necessary libraries by running:

```bash
pip install pandas scikit-learn imbalanced-learn numpy
```

## How to Run
Clone this repository or download the script diabetes_prediction_naive_bayes.py.

Make sure the required libraries are installed.

## Run the script:
```
bash
python diabetes_prediction_naive_bayes.py
```
## Output
The script will print the following outputs:

- Cross-validation scores: The 5-fold cross-validation scores to help assess the model’s generalization.

- Test accuracy: The accuracy of the model on the test set.

- Classification report: Precision, recall, F1-score, and support for each class.

- Additional performance metrics: Precision of the model.