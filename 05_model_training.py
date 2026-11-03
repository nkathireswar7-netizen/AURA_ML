import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ==========================================
# AURA - STAGE 5: MODEL TRAINING
# ==========================================

X_TRAIN_FILE = "X_train.csv"
Y_TRAIN_FILE = "y_train.csv"
MODEL_FOLDER = "models"
MODEL_FILE = "aura_failure_model.joblib"

print("========================================")
print("       AURA MODEL TRAINING")
print("========================================")

# ------------------------------------------
# 1. Load training data
# ------------------------------------------

X_train = pd.read_csv(X_TRAIN_FILE)
y_train = pd.read_csv(Y_TRAIN_FILE).squeeze()

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

# ------------------------------------------
# 2. Create Random Forest model
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nRandom Forest Configuration:")
print("Number of Trees:", 200)
print("Random State:", 42)
print("Class Weight: balanced")

# ------------------------------------------
# 3. Train model
# ------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training Completed Successfully.")

# ------------------------------------------
# 4. Display feature importance
# ------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

# ------------------------------------------
# 5. Create model folder
# ------------------------------------------

os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------------------------------
# 6. Save trained model
# ------------------------------------------

model_path = os.path.join(
    MODEL_FOLDER,
    MODEL_FILE
)

joblib.dump(model, model_path)

print("\n========================================")
print("       MODEL TRAINING COMPLETED")
print("========================================")

print("\nModel Saved:")
print(model_path)