# src/data/normalize_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Paths
DATA_DIR = "data/processed_data"
X_train_path = os.path.join(DATA_DIR, "X_train.csv")
X_test_path = os.path.join(DATA_DIR, "X_test.csv")
X_train_scaled_path = os.path.join(DATA_DIR, "X_train_scaled.csv")
X_test_scaled_path = os.path.join(DATA_DIR, "X_test_scaled.csv")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Create model directory if needed
os.makedirs("models", exist_ok=True)

# Load the data
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled data
pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(X_train_scaled_path, index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(X_test_scaled_path, index=False)

# Save the scaler for future use (e.g. prediction pipeline)
joblib.dump(scaler, SCALER_PATH)

print("✅ Normalization complete and saved.")
