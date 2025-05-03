# Diabetes Prediction using Neural Network

This project demonstrates the use of a neural network to predict diabetes based on the Pima Indians Diabetes dataset. The goal is to classify whether a person has diabetes (binary classification) using various medical features.

## Dataset

The dataset used for this project is the **Pima Indians Diabetes Database**, which can be found on Kaggle. The dataset contains several medical measurements along with a target variable indicating whether a person has diabetes or not.

- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI (Body Mass Index)
  - DiabetesPedigreeFunction
  - Age
  
- **Target**:
  - Outcome (0 = No diabetes, 1 = Diabetes)

- **Source**: [Pima Indians Diabetes Database on Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Steps

1. **Load and Preprocess the Dataset**:
   - Load the dataset from a CSV file.
   - Split the data into features (X) and target (y).
   - Split the data into training and test sets.
   - Standardize the feature data using `StandardScaler` to improve model performance.

2. **Build the Neural Network Model**:
   - The model consists of two hidden layers with ReLU activation and a final output layer with a sigmoid activation function for binary classification.

3. **Train the Model**:
   - The model is trained using the Adam optimizer and binary cross-entropy loss function for 30 epochs.
   - A validation split of 20% is used during training.

4. **Evaluate the Model**:
   - The model is evaluated on the test set, and the accuracy is printed.

5. **Plot Training and Validation Metrics**:
   - Plots for both training and validation accuracy and loss are generated and saved.

6. **Save the Model**:
   - The trained model is saved as a `.h5` file for later use.

## Files and Outputs

- **Plots**:
  - `training_validation_accuracy_plot.png`: Shows the training and validation accuracy over the epochs.
  - `training_validation_loss_plot.png`: Shows the training and validation loss over the epochs.

- **Model**:
  - `diabetes_nn_model.h5`: The saved trained neural network model.

## Requirements

To run the script, you need the following libraries:

- `pandas`
- `numpy`
- `tensorflow` (for building and training the neural network)
- `scikit-learn` (for data preprocessing and splitting)
- `matplotlib` (for plotting)

You can install the required libraries by running:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```




###  Run the script:
```bash
python diabetes_prediction_nn.py
```

After running the script, the plots will be saved in the current directory, and the trained model will be saved as diabetes_nn_model.h5.

### Evaluation
- The performance of the model can be evaluated by checking the accuracy of the test set.
- The training and validation accuracy, as well as the loss, can be observed from the generated plots.

