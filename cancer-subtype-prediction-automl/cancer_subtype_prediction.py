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

