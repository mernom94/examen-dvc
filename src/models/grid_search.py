# src/models/grid_search.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

# Paths
DATA_DIR = "data/processed_data"
MODEL_DIR = "models"
X_train_path = os.path.join(DATA_DIR, "X_train_scaled.csv")
y_train_path = os.path.join(DATA_DIR, "y_train.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# Create model directory if needed
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()  # convert from DataFrame to Series if needed

# Define model and parameter grid
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save best model
joblib.dump(grid_search.best_estimator_, BEST_MODEL_PATH)

print("✅ Grid search complete. Best model saved.")
print("Best parameters:", grid_search.best_params_)
