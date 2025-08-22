import os
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn
import json

# Set the MLflow experiment
mlflow.set_experiment("My_Linear_Model_Experiment")

# Explicitly set MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"

# Load parameters from JSON file
with open('params.json', 'r') as f:
    params = json.load(f)

n_samples = params.get("n_samples", 100)
model_type = params.get("model_type", "LinearRegression")
fit_intercept = params.get("fit_intercept", True)

# Generate some dummy data based on n_samples
X = np.random.rand(n_samples, 1) * 10
y = 2 * X + 1 + np.random.randn(n_samples, 1) * 2

with mlflow.start_run():
    # Create and train the model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)

    # Log parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("n_samples", n_samples)
    mlflow.log_param("fit_intercept", fit_intercept)

    # Log a dummy metric (replace with actual metrics later)
    mlflow.log_metric("dummy_metric", 0.95)

    # Save the model
    joblib.dump(model, 'linear_regression_model.pkl')
    mlflow.log_artifact('linear_regression_model.pkl')

    print("Linear Regression Model trained and saved with MLflow tracking.")
