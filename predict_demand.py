
from pathlib import Path
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "outputs" / "shop_demand_model.joblib"
DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "shop_business_analysis.csv"
)

# Check required files
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python train_model.py first."
    )

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Analysis dataset not found. Run python analyze_data.py first."
    )

# Load the trained model and data
model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

neighbourhoods = sorted(df["Neighbourhood"].dropna().unique())
categories = sorted(df["Shop_Category"].dropna().unique())

print("\n===== BUSINESS DEMAND PREDICTION =====")
print("\nAvailable neighbourhoods:")
for i, name in enumerate(neighbourhoods, start=1):
    print(f"{i}. {name}")

try:
    location_number = int(input("\nChoose neighbourhood number: "))

    if not 1 <= location_number <= len(neighbourhoods):
        raise ValueError("Invalid neighbourhood number.")

    neighbourhood = neighbourhoods[location_number - 1]

    print("\nAvailable shop categories:")
    for i, name in enumerate(categories, start=1):
        print(f"{i}. {name}")

    category_number = int(input("\nChoose shop category number: "))

    if not 1 <= category_number <= len(categories):
        raise ValueError("Invalid shop category number.")

    category = categories[category_number - 1]

    population = int(input("Enter estimated population of the area: "))
    competitors = int(input("Enter number of competing shops: "))
    monthly_rent = float(input("Enter expected monthly rent (INR): "))

    if population <= 0:
        raise ValueError("Population must be greater than zero.")

    if competitors < 0:
        raise ValueError("Competitor count cannot be negative.")

    if monthly_rent < 0:
        raise ValueError("Rent cannot be negative.")

    # Match the features used during model training
    new_business = pd.DataFrame([{
        "Neighbourhood": neighbourhood,
        "Shop_Category": category,
        "Population": population,
        "Competitors": competitors,
        "Monthly_Rent_INR": monthly_rent,
    }])

    # Predict demand
    prediction = float(model.predict(new_business)[0])

    # Transaction count cannot be negative
    estimated_transactions = max(0, round(prediction))

    print("\n===== PREDICTION RESULT =====")
    print("Neighbourhood:", neighbourhood)
    print("Shop category:", category)
    print("Estimated monthly transactions:", estimated_transactions)

    print(
        "\nThis is a model estimate, not a guaranteed number "
        "of customers or sales."
    )
    print(
        "The model uses a small illustrative dataset. "
        "Validate with real local demand data before investing."
    )

except ValueError as error:
    print("\nInput error:", error)
    print("Please run the program again and enter valid values.")
