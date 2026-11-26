
from pathlib import Path
import joblib
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================
PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "shop_business_analysis.csv"
)

MODEL_FILE = (
    PROJECT_DIR / "outputs" / "shop_demand_model.joblib"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Analysis data not found. Run analyze_data.py first."
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================
required_columns = [
    "Neighbourhood",
    "Shop_Category",
    "Population",
    "Competitors",
    "Monthly_Rent_INR",
    "Monthly_Demand_Transactions",
    "Expected_Monthly_Revenue_INR",
    "Monthly_COGS_INR",
    "Monthly_Salaries_INR",
    "Monthly_Utilities_Other_INR",
    "Monthly_Marketing_Other_INR",
    "Initial_Investment_INR",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

# ==========================================
# 3. GET USER INPUTS
# ==========================================
locations = sorted(df["Neighbourhood"].dropna().unique())
categories = sorted(df["Shop_Category"].dropna().unique())

print("\n======================================")
print(" BUSINESS SUITABILITY COMPARISON")
print("======================================")

print("\nAvailable neighbourhoods:")
for i, location in enumerate(locations, 1):
    print(f"{i}. {location}")

try:
    location_number = int(input("\nChoose neighbourhood number: "))

    if not 1 <= location_number <= len(locations):
        raise ValueError("Invalid neighbourhood selection.")

    location = locations[location_number - 1]

    population = int(input("Enter estimated area population: "))
    competitors = int(input("Enter competitors for the selected shop type: "))
    rent = float(input("Enter expected monthly rent (INR): "))
    budget = float(input("Enter total investment budget (INR): "))

    if population <= 0:
        raise ValueError("Population must be greater than zero.")
    if competitors < 0:
        raise ValueError("Competitors cannot be negative.")
    if rent < 0 or budget <= 0:
        raise ValueError("Rent must be non-negative and budget positive.")

except ValueError as error:
    raise SystemExit(f"Invalid input: {error}")

# ==========================================
# 4. ESTIMATE EACH BUSINESS OPTION
# ==========================================
results = []

for category in categories:

    category_data = df[df["Shop_Category"] == category]

    # Category-level assumptions from the starter dataset
    avg_purchase_value = (
        category_data["Expected_Monthly_Revenue_INR"]
        / category_data["Monthly_Demand_Transactions"].replace(0, pd.NA)
    ).median()

    cogs_ratio = (
        category_data["Monthly_COGS_INR"]
        / category_data["Expected_Monthly_Revenue_INR"].replace(0, pd.NA)
    ).median()

    avg_salary = category_data["Monthly_Salaries_INR"].median()
    avg_utilities = category_data["Monthly_Utilities_Other_INR"].median()
    avg_marketing = category_data["Monthly_Marketing_Other_INR"].median()
    avg_investment = category_data["Initial_Investment_INR"].median()

    # Use the user-selected location and inputs
    input_data = pd.DataFrame([{
        "Neighbourhood": location,
        "Shop_Category": category,
        "Population": population,
        "Competitors": competitors,
        "Monthly_Rent_INR": rent,
    }])

    predicted_demand = max(
        0,
        float(model.predict(input_data)[0])
    )

    # Revenue and cost estimates
    estimated_revenue = predicted_demand * avg_purchase_value
    estimated_cogs = estimated_revenue * cogs_ratio

    total_monthly_cost = (
        estimated_cogs
        + rent
        + avg_salary
        + avg_utilities
        + avg_marketing
    )

    estimated_profit = estimated_revenue - total_monthly_cost

    if estimated_revenue > 0:
        profit_margin = estimated_profit / estimated_revenue * 100
    else:
        profit_margin = 0.0

    budget_status = (
        "Within budget"
        if avg_investment <= budget
        else "Over budget"
    )

    results.append({
        "Neighbourhood": location,
        "Shop_Category": category,
        "Predicted_Monthly_Transactions": round(predicted_demand),
        "Estimated_Monthly_Revenue_INR": round(estimated_revenue, 2),
        "Estimated_Monthly_Cost_INR": round(total_monthly_cost, 2),
        "Estimated_Monthly_Profit_INR": round(estimated_profit, 2),
        "Profit_Margin_Percent": round(profit_margin, 2),
        "Estimated_Initial_Investment_INR": round(avg_investment, 2),
        "Budget_Status": budget_status,
    })

# ==========================================
# 5. RANK BUSINESS OPTIONS
# ==========================================
result_df = pd.DataFrame(results)

# Only businesses within budget and with positive
# estimated operating profit qualify for the shortlist.
result_df["Suitable_For_Shortlist"] = (
    (result_df["Budget_Status"] == "Within budget")
    & (result_df["Estimated_Monthly_Profit_INR"] > 0)
)

result_df = result_df.sort_values(
    by=[
        "Suitable_For_Shortlist",
        "Estimated_Monthly_Profit_INR",
    ],
    ascending=[False, False],
).reset_index(drop=True)

result_df.insert(0, "Rank", range(1, len(result_df) + 1))

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================
print("\n======================================")
print(" BUSINESS COMPARISON RESULTS")
print("======================================")

display_columns = [
    "Rank",
    "Shop_Category",
    "Predicted_Monthly_Transactions",
    "Estimated_Monthly_Revenue_INR",
    "Estimated_Monthly_Cost_INR",
    "Estimated_Monthly_Profit_INR",
    "Estimated_Initial_Investment_INR",
    "Budget_Status",
    "Suitable_For_Shortlist",
]

print(
    result_df[display_columns].to_string(index=False)
)

# ==========================================
# 7. SAVE RESULTS
# ==========================================
output_file = OUTPUT_DIR / "business_comparison_results.csv"
result_df.to_csv(output_file, index=False)

shortlist = result_df[result_df["Suitable_For_Shortlist"]]

print("\nResults saved to:")
print(output_file)

if not shortlist.empty:
    best = shortlist.iloc[0]
    print("\nTop option for further investigation:")
    print("Shop category:", best["Shop_Category"])
    print(
        "Estimated monthly operating profit (INR):",
        round(best["Estimated_Monthly_Profit_INR"], 2),
    )
    print(
        "Estimated initial investment (INR):",
        round(best["Estimated_Initial_Investment_INR"], 2),
    )
else:
    print(
        "\nNo option currently meets both the budget "
        "and positive-profit screening criteria."
    )

print(
    "\nIMPORTANT: Revenue per transaction, cost ratios, salaries, "
    "other expenses and startup investment are derived from "
    "illustrative sample records. They are not verified market data."
)
print(
    "The ranking is a preliminary screening result, "
    "not a guarantee of business suitability or profit."
)
