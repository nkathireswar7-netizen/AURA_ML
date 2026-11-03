import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# AURA - STAGE 7: MODEL EVALUATION
# ==========================================

X_TEST_FILE = "X_test.csv"
Y_TEST_FILE = "y_test.csv"
MODEL_FILE = "models/aura_failure_model.joblib"

print("========================================")
print("       AURA MODEL EVALUATION")
print("========================================")

# ------------------------------------------
# 1. Load test data
# ------------------------------------------

X_test = pd.read_csv(X_TEST_FILE)
y_test = pd.read_csv(Y_TEST_FILE).squeeze()

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTesting Target Shape:")
print(y_test.shape)

# ------------------------------------------
# 2. Load trained model
# ------------------------------------------

model = joblib.load(MODEL_FILE)

print("\nTrained Random Forest Loaded Successfully.")

# ------------------------------------------
# 3. Generate predictions
# ------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions Generated Successfully.")

# ------------------------------------------
# 4. Calculate evaluation metrics
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# ------------------------------------------
# 5. Display metrics
# ------------------------------------------

print("\n========================================")
print("          MODEL PERFORMANCE")
print("========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

# ------------------------------------------
# 6. Classification report
# ------------------------------------------

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ------------------------------------------
# 7. Confusion matrix
# ------------------------------------------

cm = confusion_matrix(y_test, y_pred)

classes = model.classes_

confusion_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print(confusion_df)

# ------------------------------------------
# 8. Save evaluation results
# ------------------------------------------

confusion_df.to_csv(
    "AURA_confusion_matrix.csv"
)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_df.to_csv(
    "AURA_model_metrics.csv",
    index=False
)

print("\n========================================")
print("       MODEL EVALUATION COMPLETED")
print("========================================")

print("\nFiles Created:")
print("- AURA_confusion_matrix.csv")
print("- AURA_model_metrics.csv")