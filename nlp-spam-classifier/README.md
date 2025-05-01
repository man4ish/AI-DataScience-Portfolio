# NLP Spam Classifier

This project is a binary text classifier that uses the Naive Bayes algorithm and TF-IDF vectorization to identify spam messages. The classifier is built using Natural Language Processing (NLP) techniques and evaluated using model performance metrics such as accuracy.

## Project Overview

In this project, we classify messages as either "spam" or "ham" (non-spam). The dataset is a collection of text messages labeled as spam or ham. We preprocess the text data, perform feature extraction using TF-IDF vectorization, and train a Naive Bayes classifier to predict whether a given message is spam.

### Tech Stack

- **pandas**: Data manipulation and cleaning.
- **scikit-learn**: Machine learning models and tools for preprocessing.
- **NLTK**: Natural Language Toolkit for text processing.
- **matplotlib**: Data visualization.
- **seaborn**: Statistical data visualization.

### Focus Areas

- **NLP Preprocessing**: Tokenization, stop-word removal, and text vectorization.
- **Vectorization**: Using TF-IDF to convert text into numerical features.
- **Model Evaluation**: Using accuracy and confusion matrix to evaluate the classifier.

## Files

- `spam.csv`: The dataset file containing labeled spam and ham messages.
- `spam_classifier.py`: The Python script for training and evaluating the spam classifier.

## Getting Started

### Prerequisites

You need to have Python 3.6 or higher installed along with the following libraries:

- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Running the Project
Clone the repository to your local machine.
```
bash
git clone <repository-url>
cd <repository-folder>
```

## Install the required dependencies.

```bash
pip install -r requirements.txt
```

## Run the classifier script to train and evaluate the spam classifier.

```bash
python spam_classifier.py
```

## The Script
- Loading Data: The dataset is loaded from a CSV file (spam.csv). The file contains two columns: label (spam/ham) and text (message content).

- Preprocessing: The text data is preprocessed by:

- Removing stop words and non-alphabetic characters.

- Tokenizing the text.

- Applying TF-IDF vectorization to convert the text into numerical format.

- Model Training: A Naive Bayes classifier is trained using the preprocessed text data.

- Evaluation: The classifier is evaluated using accuracy and confusion matrix metrics.

## Example Output
When you run the script, you should see output similar to this:

```
Accuracy: 97.20%
Confusion Matrix:
[[ 965    3]
 [  30  964]]
```

This shows the accuracy of the classifier and the confusion matrix, which helps evaluate its performance in detecting spam messages.