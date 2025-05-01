# Bioinformatics Text Classification and Prediction

This repository contains various scripts and models related to bioinformatics text processing using transformer-based models such as PubMedBERT and other neural networks for tasks such as promoter prediction, protein language model, and biomedical text generation.

## Project Structure

### Files and Directories:

- `promoter_prediction.py`:  
  A Python script for predicting promoters using a pretrained deep learning model. The model leverages sequence classification techniques to classify DNA sequences as promoters or non-promoters.

- `hf_env/`:  
  This directory contains the environment setup for running the Hugging Face models locally. It includes all the dependencies required for running the Python scripts, which are generally specified in a `requirements.txt` or equivalent environment setup.

- `protein_language_model.py`:  
  A Python script that implements protein sequence modeling using transformer-based architectures. The model is designed to learn protein sequence embeddings for downstream bioinformatics tasks.

- `biomedical_text_generation.py`:  
  A script for generating biomedical text using pretrained transformer models. It is based on language models like PubMedBERT and is tailored for generating research abstracts or textual descriptions related to biomedical topics.

- `pubmedbert_classification_example.py`:  
  A Python script that demonstrates how to fine-tune PubMedBERT for biomedical text classification tasks. It loads a pretrained PubMedBERT model, processes biomedical text data, and outputs the model’s predictions.

## Installation

### Prerequisites:

Ensure you have Python 3.7+ installed, and the following libraries should be installed:

- `transformers` - for working with Hugging Face models.
- `torch` - for running the models.
- `numpy` - for array manipulation.
- `pandas` - for data handling (if needed).
- `biopython` - for bioinformatics-related functionalities.
- `scikit-learn` - for machine learning utilities.

You can install the required dependencies using pip:

```bash
pip install transformers torch numpy pandas biopython scikit-learn
```
Environment Setup:
To set up the environment for the Hugging Face models, use the following command:

```bash
cd hf_env
pip install -r requirements.txt
```
Alternatively, you can create a new virtual environment and install the required dependencies:

```bash
python -m venv hf_env
source hf_env/bin/activate
pip install -r requirements.txt
```
# Usage
## 1. Promoter Prediction:
Run promoter_prediction.py to predict whether a given DNA sequence is a promoter or not.

```bash
python promoter_prediction.py
```
## 2. Protein Language Model:
Run protein_language_model.py to generate protein embeddings from protein sequences using transformer-based models.

```bash
python protein_language_model.py
```
## 3. Biomedical Text Generation:
Run biomedical_text_generation.py to generate biomedical text using a pretrained language model.

```bash
python biomedical_text_generation.py
```
## 4. PubMedBERT Classification Example:
Run pubmedbert_classification_example.py to classify biomedical text into categories using a fine-tuned PubMedBERT model.

```bash
python pubmedbert_classification_example.py
```
