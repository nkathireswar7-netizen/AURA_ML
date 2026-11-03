import pandas as pd

# ==========================================
# AURA - STAGE 3: FEATURE ENGINEERING
# ==========================================

INPUT_FILE = "AURA_failure_cleaned.csv"
OUTPUT_FILE = "AURA_failure_features.csv"

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("========================================")
print("       AURA FEATURE ENGINEERING")
print("========================================")

print("\nOriginal Shape:")
print(df.shape)

# ------------------------------------------
# 1. Battery Risk
# ------------------------------------------

df["battery_risk"] = 100 - df["battery"]

# ------------------------------------------
# 2. Obstacle Risk
# ------------------------------------------

df["obstacle_risk"] = 1 / (df["obstacle_distance"] + 1)

# ------------------------------------------
# 3. GPS + Communication Risk
# ------------------------------------------

df["communication_gps_risk"] = (
    (1 - df["gps_signal"]) *
    (1 - df["communication"])
)

# ------------------------------------------
# 4. Mission Remaining
# ------------------------------------------

df["mission_remaining"] = 100 - df["mission_progress"]

# ------------------------------------------
# 5. Battery / Home Distance Ratio
# ------------------------------------------

df["battery_home_ratio"] = (
    df["battery"] /
    (df["distance_home"] + 1)
)

# ------------------------------------------
# Display engineered features
# ------------------------------------------

engineered_features = [
    "battery_risk",
    "obstacle_risk",
    "communication_gps_risk",
    "mission_remaining",
    "battery_home_ratio"
]

print("\nNew Engineered Features:")

for feature in engineered_features:
    print("-", feature)

print("\nNew Dataset Shape:")
print(df.shape)

print("\nFeature Preview:")
print(df[engineered_features].head())

# ------------------------------------------
# Save feature-engineered dataset
# ------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("   FEATURE ENGINEERING COMPLETED")
print("========================================")

print("\nSaved File:")
print(OUTPUT_FILE)