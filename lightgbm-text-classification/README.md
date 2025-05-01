# LightGBM Text Classification

This project demonstrates how to use LightGBM (Light Gradient Boosting Machine) for text classification tasks. It uses a dataset of text messages, and the goal is to classify them as either "spam" or "ham" using a LightGBM model.

## Overview

In this project, we:
1. Preprocess the text data.
2. Convert the text data into numeric form using TF-IDF vectorization.
3. Train a LightGBM classifier.
4. Evaluate the model's performance using accuracy and confusion matrix.

## Tech Stack

- **LightGBM**: Gradient boosting framework for faster training and better performance.
- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Model training and evaluation, including TF-IDF vectorization.
- **nltk**: Natural Language Processing (NLP) utilities.
- **matplotlib & seaborn**: Data visualization and performance evaluation.

## Steps

1. **Data Preprocessing**: 
    - The dataset is loaded from a CSV file.
    - Text data is cleaned and preprocessed, such as removing stop words and non-alphabetical characters.
    
2. **Vectorization**: 
    - TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text data into numerical form.

3. **Model Training**:
    - LightGBM is used to train a classification model on the TF-IDF features.

4. **Model Evaluation**: 
    - The model's performance is evaluated using accuracy and confusion matrix.
    
## Installation

To run this project, you'll need to install the required dependencies.

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/lightGBM-text-classification.git
    cd lightGBM-text-classification
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your dataset**: Ensure that your dataset is a CSV file with a 'label' column (indicating spam/ham) and a 'text' column (containing the message text).

2. **Run the script**: Execute the script to train the model and evaluate performance.
    ```bash
    python lightgbm_text_classification.py
    ```

## Results

- The script will output the model’s accuracy and display a confusion matrix.
- You can adjust parameters such as the number of boosting rounds or the learning rate for better performance.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and create a pull request with your improvements or fixes.

## Author

Manish Kumar
Email: mandecent.gupta@gmail.com  
GitHub: [man4ish](https://github.com/man4ish)
