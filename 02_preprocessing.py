import pandas as pd

# ==========================================
# AURA - STAGE 2: PREPROCESSING & CLEANING
# ==========================================

DATA_FILE = "AURA_ML_Training_Dataset_2000 (1).csv"
OUTPUT_FILE = "AURA_failure_cleaned.csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

print("========================================")
print("    AURA PREPROCESSING & CLEANING")
print("========================================")

print("\nOriginal Shape:")
print(df.shape)

# ------------------------------------------
# 1. Check duplicate rows
# ------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# ------------------------------------------
# 2. Check missing values
# ------------------------------------------

missing_values = df.isnull().sum().sum()

print("\nTotal Missing Values:", missing_values)

if missing_values > 0:
    print("Missing values detected.")
else:
    print("No missing values found.")

# ------------------------------------------
# 3. Check infinite values
# ------------------------------------------

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

infinite_values = (
    df[numeric_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("\nInfinite Values:", infinite_values)

# ------------------------------------------
# 4. Check target values
# ------------------------------------------

print("\nFailure Classes:")
print(df["failure_type"].unique())

# ------------------------------------------
# 5. Validate telemetry ranges
# ------------------------------------------

print("\nTelemetry Range Check:")

print("Battery:")
print("Minimum:", df["battery"].min())
print("Maximum:", df["battery"].max())

print("\nGPS Signal Values:")
print(df["gps_signal"].unique())

print("\nCommunication Values:")
print(df["communication"].unique())

print("\nObstacle Distance:")
print("Minimum:", df["obstacle_distance"].min())

print("\nMission Progress:")
print("Minimum:", df["mission_progress"].min())
print("Maximum:", df["mission_progress"].max())

# ------------------------------------------
# 6. Save cleaned dataset
# ------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("    PREPROCESSING COMPLETED")
print("========================================")

print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nSaved File:")
print(OUTPUT_FILE)