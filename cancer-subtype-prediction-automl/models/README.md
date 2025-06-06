# Cancer Subtype Prediction using H2O AutoML

This project uses H2O.ai AutoML to build a machine learning model that predicts breast cancer subtypes from transcriptomic gene expression data.

---

## Features

- Automated ML pipeline using H2O AutoML  
- Classification of cancer subtypes using gene expression profiles  
- Compares multiple models (GLM, GBM, Deep Learning, etc.)  
- Saves best model  
- Easy to extend to TCGA or METABRIC data  

---

## Dataset

Simulated RNA-seq data with subtype labels.

Example:

```csv
SampleID,BRCA1,ESR1,ERBB2,FOXA1,MYC,Subtype
S1,8.5,10.2,2.1,9.8,5.5,Luminal A
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o
```

### 2. Run Script

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O server
h2o.init()

# Load dataset (replace with your CSV path)
data = h2o.import_file("breast_cancer_expression.csv")

# Convert target to categorical
data["Subtype"] = data["Subtype"].asfactor()

# Define features and target
x = data.columns[1:-1]  # all gene columns except SampleID and Subtype
y = "Subtype"

# Split data into train and test
train, test = data.split_frame(ratios=[0.8], seed=42)

# Run H2O AutoML for 1 minute or max 10 models
aml = H2OAutoML(max_runtime_secs=60, max_models=10, seed=1, verbosity="info")
aml.train(x=x, y=y, training_frame=train)

# View the leaderboard
print("Leaderboard:")
print(aml.leaderboard.head(rows=10))

# Get the best model
leader = aml.leader
print(f"\nBest model ID: {leader.model_id}")

# Variable importance (if supported)
if hasattr(leader, "varimp"):
    print("\nVariable Importance:")
    vi = leader.varimp(use_pandas=True)
    print(vi)

# Save the leader model
model_path = h2o.save_model(model=leader, path="models/", force=True)
print(f"\nModel saved to: {model_path}")

# Predict on test set
preds = leader.predict(test)
print("\nSample Predictions:")
print(preds.head())

# Shutdown H2O (optional)
# h2o.shutdown(prompt=False)
```

---

## Output

- Leaderboard with models ranked by performance  
- Best model saved in `models/`  
- Predictions on test data  

---

## Tools

- H2O.ai AutoML  
- Python  
- Gene expression dataset  

