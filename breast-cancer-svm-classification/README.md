# Breast Cancer Classification using SVM

This repository contains a Python implementation of a breast cancer classification model using Support Vector Machine (SVM) on a publicly available dataset. The model is trained to classify tumor samples as either **malignant** or **benign**.

## Objective

The main goal of this project is to demonstrate the application of **Support Vector Machines (SVM)** in a real-world scenario for binary classification. The dataset used in this project contains information about various features of cell nuclei present in breast cancer biopsies, and the task is to classify the tumors as either malignant or benign.

## Dataset

The dataset used is from the **UCI Machine Learning Repository** and is the **Breast Cancer Wisconsin (Diagnostic) dataset**. It contains 569 instances, with 30 features for each sample, including:
- **Radius**
- **Texture**
- **Perimeter**
- **Area**
- **Smoothness**
- **Compactness**
- **Concavity**
- **Concave points**
- **Symmetry**
- **Fractal dimension**

### Classes:
- **Malignant**: Tumors that are cancerous.
- **Benign**: Tumors that are non-cancerous.

## Dependencies

To run this project, ensure that you have the following Python libraries installed:

- `scikit-learn`
- `numpy`
- `pandas`
- `matplotlib`

You can install them using `pip`:

```bash
pip install scikit-learn numpy pandas matplotlib
```

## SVM Classification Model
The project uses Support Vector Machine (SVM) for classification. The following steps were followed:

- Loading the dataset: The dataset was loaded using scikit-learn's load_breast_cancer function.

- Data Preprocessing: We scaled the features using StandardScaler to improve the performance of the SVM model.

- Model Training: A linear kernel SVM was used to train the model.

- Evaluation: The model was evaluated using a classification report that includes precision, recall, F1-score, and accuracy metrics.

## Run the script
```bash
python svm_breast_cancer_classification.py
```


## Output
When you run the code, it generates a classification report showing precision, recall, F1-score, and accuracy metrics for the malignant and benign classes. The output might look like this:

```markdown

              precision    recall  f1-score   support

   malignant       1.00      0.86      0.93        43
      benign       0.92      1.00      0.96        71

    accuracy                           0.95       114
   macro avg       0.96      0.93      0.94       114
weighted avg       0.95      0.95      0.95       114
```

## Interpretation:
### Precision: The proportion of true positive predictions (correct malignant or benign predictions) to the total predicted positives.

- For malignant tumors: 1.00 (perfect precision).

- For benign tumors: 0.92 (92% of benign predictions were correct).

### Recall: The proportion of true positives to the actual positives (i.e., how many of the actual malignant or benign tumors were correctly identified).

- For malignant tumors: 0.86 (86% of actual malignant tumors were correctly identified).

- For benign tumors: 1.00 (100% of actual benign tumors were correctly identified).

### F1-Score: The harmonic mean of precision and recall.

- For malignant tumors: 0.93.

- For benign tumors: 0.96.

- Accuracy: 95% — The overall percentage of correct predictions.

## Macro Average: The average of precision, recall, and F1-score across both classes, treating each class equally.

- Precision: 0.96, Recall: 0.93, F1-Score: 0.94.

## Weighted Average: The average precision, recall, and F1-score weighted by the support (class distribution).

Precision: 0.95, Recall: 0.95, F1-Score: 0.95.

## Conclusion
The SVM model has shown excellent performance in classifying breast cancer tumors as either malignant or benign.

The accuracy of 95% indicates that the model is making reliable predictions.

The classification report gives insight into the precision and recall for both classes, making it clear that the model is well-balanced in detecting both benign and malignant tumors.

