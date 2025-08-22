import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import mlflow
import mlflow.sklearn
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set the MLflow experiment
mlflow.set_experiment("My_Linear_Model_Experiment")

# Explicitly set MLFLOW_TRACKING_URI
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow-server:5000"

# Load parameters from JSON file
with open('params.json', 'r') as f:
    params = json.load(f)

model_type = params.get("model_type", "LinearRegression")
fit_intercept = params.get("fit_intercept", True)

# ---------------- LOAD CSV FILE ----------------
CSV_PATH = params.get("csv_path", "Machine_Load_vs_Power_Consumption_Dataset.csv")
df = pd.read_csv(CSV_PATH)

X = df[['machine_load']].values
y = df['power_consumption'].values
# ------------------------------------------------

with mlflow.start_run():
    # Create and train the model
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    # Log parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("fit_intercept", fit_intercept)
    mlflow.log_param("csv_path", CSV_PATH)

    # Log real metrics
    mlflow.log_metric("mae", mean_absolute_error(y, y_pred))
    mlflow.log_metric("rmse", mean_squared_error(y, y_pred, squared=False))
    mlflow.log_metric("r2", r2_score(y, y_pred))

    # Save the model
    joblib.dump(model, 'linear_regression_model.pkl')
    mlflow.log_artifact('linear_regression_model.pkl')

    print("Model trained and saved with MLflow tracking.")
