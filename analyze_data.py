
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "shop_business_starter_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check whether the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Please copy the CSV into data/raw."
    )

# Load the dataset
df = pd.read_csv(DATA_FILE)

print("\n========== DATASET INFORMATION ==========")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate rows:", df.duplicated().sum())

# Calculate total monthly operating costs
cost_columns = [
    "Monthly_COGS_INR",
    "Monthly_Rent_INR",
    "Monthly_Salaries_INR",
    "Monthly_Utilities_Other_INR",
    "Monthly_Marketing_Other_INR",
]

required_columns = cost_columns + [
    "Expected_Monthly_Revenue_INR",
    "Initial_Investment_INR",
    "Monthly_Demand_Transactions",
    "Neighbourhood",
    "Shop_Category",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Calculate costs and estimated operating profit
df["Total_Monthly_Operating_Cost_INR"] = (
    df[cost_columns].sum(axis=1)
)

df["Estimated_Monthly_Net_Operating_Profit_INR"] = (
    df["Expected_Monthly_Revenue_INR"]
    - df["Total_Monthly_Operating_Cost_INR"]
)

# Calculate profit margin safely
df["Profit_Margin_Percent"] = (
    df["Estimated_Monthly_Net_Operating_Profit_INR"]
    .div(df["Expected_Monthly_Revenue_INR"].where(
        df["Expected_Monthly_Revenue_INR"] != 0
    ))
    .mul(100)
    .round(2)
)

# Rank business options by estimated operating profit
df["Profit_Rank"] = (
    df.groupby("Neighbourhood")[
        "Estimated_Monthly_Net_Operating_Profit_INR"
    ]
    .rank(method="dense", ascending=False)
    .astype(int)
)

# Save the analysed dataset
output_file = OUTPUT_DIR / "shop_business_analysis.csv"
df.to_csv(output_file, index=False)

# Display business comparison
print("\n========== BUSINESS PROFIT COMPARISON ==========")

summary_columns = [
    "Neighbourhood",
    "Shop_Category",
    "Monthly_Demand_Transactions",
    "Expected_Monthly_Revenue_INR",
    "Total_Monthly_Operating_Cost_INR",
    "Estimated_Monthly_Net_Operating_Profit_INR",
    "Profit_Margin_Percent",
    "Initial_Investment_INR",
    "Profit_Rank",
]

print(
    df[summary_columns]
    .sort_values(
        ["Neighbourhood", "Profit_Rank"]
    )
    .to_string(index=False)
)

print("\nAnalysis completed successfully.")
print("Saved results to:", output_file)
print(
    "\nWARNING: These are illustrative calculations, "
    "not verified real-world profit predictions."
)
