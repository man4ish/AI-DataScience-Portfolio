"""
NLP Spam Classifier

This module builds a binary text classifier to detect spam messages using natural language processing techniques.
It uses a dataset of SMS messages labeled as 'spam' or 'ham' and applies the following steps:

1. Data Loading and Preprocessing
2. TF-IDF Vectorization
3. Training a Multinomial Naive Bayes classifier
4. Evaluating performance using accuracy and confusion matrix

Tech Stack:
- pandas: Data loading and manipulation
- NLTK: Tokenization, stopword removal
- scikit-learn: Vectorization, modeling, and evaluation
- seaborn & matplotlib: Visualization

Author: Manish Kumar
Date: 2025-05-31
"""

import pandas as pd
import nltk
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()

def preprocess_text(text):
    """
    Perform text preprocessing including:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Stemming
    
    Args:
        text (str): The original message

    Returns:
        str: Cleaned and preprocessed text
    """
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def main():
    # Load data
    # Load the CSV file and inspect the column names
    df = pd.read_csv('data/spam.csv', encoding='latin-1')

    # Check the columns in the loaded CSV to ensure correct column names
    print(df.columns)

    # Now, select only the necessary columns and rename them if needed
    df = df[['label', 'text']]  # Replace 'v1' and 'v2' with 'label' and 'text' if required
    df.columns = ['label', 'message']  # Rename the columns to 'label' and 'text'

    print(df['message'])
    # Preprocess messages
    df['cleaned'] = df['message'].apply(preprocess_text)

    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Train classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()
