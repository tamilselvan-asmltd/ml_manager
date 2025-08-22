import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow
import mlflow.sklearn
import json

# Set the MLflow experiment
mlflow.set_experiment("My_Logistic_Model_Experiment")

# Explicitly set MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"

# Load parameters from JSON file
with open('params.json', 'r') as f:
    params = json.load(f)

n_samples = params.get("n_samples", 200)
model_type = params.get("model_type", "LogisticRegression")
solver = params.get("solver", "liblinear")

# Generate some dummy data based on n_samples
X = np.random.rand(n_samples, 1) * 10
y = np.random.randint(0, 2, n_samples)

with mlflow.start_run():
    # Create and train the model
    model = LogisticRegression(solver=solver)
    model.fit(X, y)

    # Log parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("solver", solver)

    # Log a dummy metric (replace with actual metrics later)
    mlflow.log_metric("dummy_metric", 0.88)

    # Save the model
    joblib.dump(model, 'logistic_regression_model.pkl')
    mlflow.log_artifact('logistic_regression_model.pkl')

    print("Logistic Regression Model trained and saved with MLflow tracking.")
