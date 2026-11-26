
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -------------------------------------
# 1. Set project paths
# -------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "shop_business_analysis.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "shop_demand_model.joblib"
METRICS_FILE = OUTPUT_DIR / "model_evaluation.txt"

# -------------------------------------
# 2. Load dataset
# -------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run python analyze_data.py first."
    )

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# -------------------------------------
# 3. Select features and target
# -------------------------------------
categorical_features = [
    "Neighbourhood",
    "Shop_Category",
]

numerical_features = [
    "Population",
    "Competitors",
    "Monthly_Rent_INR",
]

target = "Monthly_Demand_Transactions"

required_columns = (
    categorical_features
    + numerical_features
    + [target]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Keep only the required columns and remove incomplete rows
model_df = df[required_columns].copy()

for column in numerical_features + [target]:
    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce",
    )

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan,
).dropna()

if len(model_df) < 10:
    raise ValueError(
        "Not enough complete records to train and test the model."
    )

X = model_df[
    categorical_features + numerical_features
]

y = model_df[target]

# -------------------------------------
# 4. Split training and testing data
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# -------------------------------------
# 5. Prepare categorical features
# -------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numbers",
            "passthrough",
            numerical_features,
        ),
    ]
)

# -------------------------------------
# 6. Create Random Forest model
# -------------------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    ]
)

# -------------------------------------
# 7. Train the model
# -------------------------------------
print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed.")

# -------------------------------------
# 8. Evaluate on unseen test data
# -------------------------------------
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  {mae:.2f} transactions")
print(f"RMSE: {rmse:.2f} transactions")
print(f"R2:   {r2:.3f}")

# Display actual versus predicted values
comparison = pd.DataFrame({
    "Actual_Demand": y_test.to_numpy(),
    "Predicted_Demand": np.round(predictions, 2),
})

comparison["Absolute_Error"] = (
    comparison["Actual_Demand"]
    - comparison["Predicted_Demand"]
).abs()

print("\nActual vs predicted demand:")
print(comparison.to_string(index=False))

# -------------------------------------
# 9. Save the trained pipeline
# -------------------------------------
joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

# -------------------------------------
# 10. Save evaluation results
# -------------------------------------
with open(METRICS_FILE, "w", encoding="utf-8") as file:
    file.write("Random Forest Demand Model Evaluation\n")
    file.write("=====================================\n")
    file.write(f"Dataset rows used: {len(model_df)}\n")
    file.write(f"Training rows: {len(X_train)}\n")
    file.write(f"Testing rows: {len(X_test)}\n")
    file.write(f"MAE: {mae:.2f}\n")
    file.write(f"RMSE: {rmse:.2f}\n")
    file.write(f"R2: {r2:.3f}\n")
    file.write(
        "\nWarning: Evaluation uses a small illustrative dataset. "
        "Metrics do not establish real-world accuracy.\n"
    )

print("\nEvaluation saved to:")
print(METRICS_FILE)

print("\nStep 7 completed successfully.")
