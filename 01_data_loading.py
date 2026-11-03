import pandas as pd

# ==========================================
# AURA - STAGE 1: DATA LOADING
# ==========================================

DATA_FILE = "AURA_ML_Training_Dataset_2000 (1).csv"

# Load dataset
df = pd.read_csv(DATA_FILE)

print("========================================")
print("       AURA DATA LOADING STAGE")
print("========================================")

# Dataset size
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumns:")
for column in df.columns:
    print("-", column)

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Target distribution
print("\nFailure Type Distribution:")
print(df["failure_type"].value_counts())

print("\n========================================")
print("       DATA LOADING COMPLETED")
print("========================================")