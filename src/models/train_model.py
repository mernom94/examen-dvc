# src/models/train_model.py

import pandas as pd
import joblib
import os

# Paths
DATA_DIR = "data/processed_data"
MODEL_DIR = "models"
X_train_path = os.path.join(DATA_DIR, "X_train_scaled.csv")
y_train_path = os.path.join(DATA_DIR, "y_train.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")

# Create model directory if needed
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze()

# Load best model (from GridSearch)
model = joblib.load(BEST_MODEL_PATH)

# Train the model on full training data
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, FINAL_MODEL_PATH)

print("✅ Model training complete. Final model saved.")
