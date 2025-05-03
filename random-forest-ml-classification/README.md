# Random Forest Classification - Iris Dataset

This project demonstrates the application of the Random Forest algorithm for classifying the Iris dataset, a well-known dataset in machine learning. The goal of this project is to predict the species of iris flowers based on various features such as sepal length, sepal width, petal length, and petal width.

## Features:
- **Training and Testing**: The dataset is split into training and testing sets to evaluate the model's performance.
- **Cross-Validation**: 5-fold cross-validation is implemented to ensure more reliable performance estimation.
- **Hyperparameter Tuning**: Hyperparameter optimization using GridSearchCV to find the best model configuration.
- **Model Evaluation**: Model performance is evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

## Steps Involved:
1. **Data Loading and Preprocessing**: The Iris dataset is loaded, and the data is preprocessed to handle any potential issues.
2. **Model Training**: A Random Forest model is trained on the training data.
3. **Cross-Validation**: Model performance is evaluated using k-fold cross-validation.
4. **Hyperparameter Tuning**: GridSearchCV is used to find the optimal hyperparameters for the Random Forest model.
5. **Model Evaluation**: The model's accuracy and classification report (precision, recall, and F1-score) are displayed on the test set.
6. **Result Reporting**: The results are printed, showing the final model's performance, cross-validation accuracy, and classification metrics.

## Requirements:
- Python 3.x
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```