
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "shop_business_analysis.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"File not found: {DATA_FILE}\n"
        "Run python analyze_data.py first."
    )

df = pd.read_csv(DATA_FILE)

# Required columns
required_columns = [
    "Neighbourhood",
    "Shop_Category",
    "Monthly_Demand_Transactions",
    "Expected_Monthly_Revenue_INR",
    "Estimated_Monthly_Net_Operating_Profit_INR",
    "Competitors",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

sns.set_theme(style="whitegrid")

# ---------------------------------------
# Chart 1: Average estimated profit
# by shop category
# ---------------------------------------
profit = (
    df.groupby("Shop_Category")[
        "Estimated_Monthly_Net_Operating_Profit_INR"
    ]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
profit.plot(kind="bar")
plt.axhline(y=0, linestyle="--")
plt.title("Average Estimated Monthly Profit by Shop Category")
plt.xlabel("Shop Category")
plt.ylabel("Average Estimated Profit (INR)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "average_profit_by_category.png",
    dpi=150
)
plt.close()

# ---------------------------------------
# Chart 2: Estimated demand by category
# ---------------------------------------
demand = (
    df.groupby("Shop_Category")[
        "Monthly_Demand_Transactions"
    ]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
demand.plot(kind="bar")
plt.title("Average Monthly Demand by Shop Category")
plt.xlabel("Shop Category")
plt.ylabel("Monthly Transactions")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "demand_by_category.png",
    dpi=150
)
plt.close()

# ---------------------------------------
# Chart 3: Revenue vs operating costs
# ---------------------------------------
plt.figure(figsize=(9, 6))

plt.scatter(
    df["Expected_Monthly_Revenue_INR"],
    df["Total_Monthly_Operating_Cost_INR"],
    alpha=0.8
)

plt.title("Estimated Monthly Revenue vs Operating Costs")
plt.xlabel("Revenue (INR)")
plt.ylabel("Operating Costs (INR)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "revenue_vs_costs.png",
    dpi=150
)
plt.close()

# ---------------------------------------
# Chart 4: Profit by neighbourhood
# ---------------------------------------
neighbourhood_profit = (
    df.groupby("Neighbourhood")[
        "Estimated_Monthly_Net_Operating_Profit_INR"
    ]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
neighbourhood_profit.plot(kind="bar")
plt.axhline(y=0, linestyle="--")
plt.title("Average Estimated Monthly Profit by Neighbourhood")
plt.xlabel("Neighbourhood")
plt.ylabel("Average Estimated Profit (INR)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "profit_by_neighbourhood.png",
    dpi=150
)
plt.close()

# ---------------------------------------
# Chart 5: Competitors vs demand
# ---------------------------------------
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Competitors",
    y="Monthly_Demand_Transactions",
    hue="Shop_Category",
    style="Shop_Category",
    s=100
)

plt.title("Competitor Count vs Monthly Demand")
plt.xlabel("Number of Competitors")
plt.ylabel("Monthly Transactions")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "competitors_vs_demand.png",
    dpi=150
)
plt.close()

print("\nAll five charts were created successfully.")

print("\nCharts saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)

print(
    "\nNote: These charts use illustrative data. "
    "They do not establish actual market demand or profitability."
)
