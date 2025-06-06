# Cancer Subtype Prediction using H2O AutoML

This project demonstrates the use of **H2O.ai AutoML** to predict breast cancer subtypes based on transcriptomic gene expression data. Using real-world-like RNA-seq data, the pipeline automatically trains and evaluates multiple machine learning models to identify the best performing model for classifying cancer subtypes.

---

## Features

- Automated model training and selection using H2O AutoML  
- Supports multiple classification algorithms (GLM, GBM, Deep Learning, etc.)  
- Generates model leaderboard with detailed performance metrics  
- Provides variable importance for feature interpretability  
- Saves the best model for downstream use  
- Easy to integrate with transcriptomics datasets in CSV format  

---

## How to Use

1. Prepare your gene expression dataset in CSV format with samples labeled by cancer subtype.  
2. Use H2O’s Python API to load data, specify features and target, and split into train/test sets.  
3. Run the AutoML process specifying runtime and model limits.  
4. Evaluate models via leaderboard and select the best model.  
5. Save the best model for future predictions.  

---

## Dependencies

- Python 3.x  
- H2O Python package (`h2o`)  
- Gene expression CSV dataset  

---

## Applications

- Cancer subtype classification from RNA-seq data  
- Biomarker discovery via variable importance analysis  
- Automated ML workflow for biomedical datasets  
