import pandas as pd
import joblib

# ==========================================
# AURA - STAGE 6: MODEL TESTING
# ==========================================

X_TEST_FILE = "X_test.csv"
Y_TEST_FILE = "y_test.csv"
MODEL_FILE = "models/aura_failure_model.joblib"

print("========================================")
print("        AURA MODEL TESTING")
print("========================================")

# 1. Load test data
X_test = pd.read_csv(X_TEST_FILE)
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# 2. Load trained model
model = joblib.load(MODEL_FILE)

print("\nTrained Random Forest Loaded Successfully.")

# 3. Make predictions
print("\nMaking Predictions...")

y_pred = model.predict(X_test)

print("Predictions Completed.")

# 4. Display predictions
print("\nFirst 20 Predictions:")

for i in range(min(20, len(y_pred))):
    print(
        f"Actual: {y_test.iloc[i]:20} "
        f"Predicted: {y_pred[i]}"
    )

# 5. Save predictions
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

results.to_csv(
    "AURA_test_predictions.csv",
    index=False
)

print("\n========================================")
print("        MODEL TESTING COMPLETED")
print("========================================")

print("\nPrediction File Saved:")
print("AURA_test_predictions.csv")