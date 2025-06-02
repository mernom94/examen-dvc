import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
RAW_DATA_PATH = "data/raw_data/raw.csv"
PROCESSED_DIR = "data/processed_data"

# Create output directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH)

# Separate features and target
features = [
    'ave_flot_air_flow',
    'ave_flot_level',
    'iron_feed',
    'starch_flow',
    'amina_flow',
    'ore_pulp_flow',
    'ore_pulp_pH',
    'ore_pulp_density'
]
target = 'silica_concentrate'

X = df[features]
y = df[target]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Save the split data
X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print("Data split and saved successfully.")