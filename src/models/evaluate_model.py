# src/models/evaluate_model.py

import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, r2_score

# Paths
DATA_DIR = "data/processed_data"
MODEL_DIR = "models"
METRICS_DIR = "metrics"
X_test_path = os.path.join(DATA_DIR, "X_test_scaled.csv")
y_test_path = os.path.join(DATA_DIR, "y_test.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "predictions.csv")
SCORES_PATH = os.path.join(METRICS_DIR, "scores.json")

# Create metrics directory if needed
os.makedirs(METRICS_DIR, exist_ok=True)

# Load data and model
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path).squeeze()
model = joblib.load(MODEL_PATH)

# Predict
y_pred = model.predict(X_test)

# Save predictions
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(PREDICTIONS_PATH, index=False)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics
metrics = {"mse": mse, "r2": r2}
with open(SCORES_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation complete.")
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
