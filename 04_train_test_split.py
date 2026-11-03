import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# AURA - STAGE 4: FEATURE SELECTION
#              & TRAIN-TEST SPLIT
# ==========================================

INPUT_FILE = "AURA_failure_features.csv"

# Load feature-engineered dataset
df = pd.read_csv(INPUT_FILE)

print("========================================")
print("     AURA TRAIN-TEST SPLIT STAGE")
print("========================================")

print("\nDataset Shape:")
print(df.shape)

# ------------------------------------------
# 1. Select input features
# ------------------------------------------

features = [
    "battery",
    "gps_signal",
    "communication",
    "obstacle_distance",
    "distance_home",
    "distance_destination",
    "speed",
    "mission_progress",
    "wind_speed",
    "altitude",
    "battery_risk",
    "obstacle_risk",
    "communication_gps_risk",
    "mission_remaining",
    "battery_home_ratio"
]

# X = input features
X = df[features]

# y = target
y = df["failure_type"]

print("\nNumber of Input Features:")
print(len(features))

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("failure_type")

# ------------------------------------------
# 2. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------
# 3. Display split information
# ------------------------------------------

print("\n========================================")
print("          SPLIT INFORMATION")
print("========================================")

print("\nTotal Samples:", len(df))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# ------------------------------------------
# 4. Save split datasets
# ------------------------------------------

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n========================================")
print("       TRAIN-TEST SPLIT COMPLETED")
print("========================================")

print("\nFiles Created:")
print("- X_train.csv")
print("- X_test.csv")
print("- y_train.csv")
print("- y_test.csv")