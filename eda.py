"""
eda.py
------
Exploratory Data Analysis for the canteen food-waste dataset.
Produces summary statistics (printed) and saved plots (in plots/) that
reveal which factors most influence food waste.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_raw_data, clean_data, engineer_features, DAY_ORDER

sns.set_theme(style="whitegrid")

df = load_raw_data()
df = clean_data(df)
df = engineer_features(df)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.info())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isna().sum())

# ---------------------------------------------------------------------
# 1. Distribution of the target variable
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["Food_Waste_KG"], bins=30, kde=True, color="teal")
plt.title("Distribution of Daily Food Waste (kg)")
plt.xlabel("Food Waste (kg)")
plt.tight_layout()
plt.savefig("plots/01_waste_distribution.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 2. Food waste by day of week
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Day", y="Food_Waste_KG", order=DAY_ORDER, palette="viridis")
plt.title("Food Waste by Day of Week")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plots/02_waste_by_day.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 3. Food waste: Holiday vs Working day
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="Holiday", y="Food_Waste_KG", palette="Set2")
plt.title("Food Waste: Holiday vs Working Day")
plt.tight_layout()
plt.savefig("plots/03_waste_by_holiday.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 4. Food waste by meal type
# ---------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="Meal_Type", y="Food_Waste_KG", palette="pastel")
plt.title("Food Waste by Meal Type")
plt.tight_layout()
plt.savefig("plots/04_waste_by_mealtype.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 5. Prepared vs Served gap vs waste (scatter)
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x="Prepared_Served_Gap", y="Food_Waste_KG",
                 hue="Special_Event", alpha=0.6)
plt.title("Unused Meals (Prepared - Served) vs Food Waste")
plt.xlabel("Meals Prepared - Meals Served")
plt.ylabel("Food Waste (kg)")
plt.tight_layout()
plt.savefig("plots/05_gap_vs_waste.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 6. Correlation heatmap of numeric features
# ---------------------------------------------------------------------
numeric_cols = ["Students_Expected", "Meals_Prepared", "Meals_Served",
                 "Previous_Waste_KG", "Avg_Waste_Last_7Days",
                 "Prepared_Served_Gap", "Is_Weekend", "Food_Waste_KG"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/06_correlation_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 7. Waste trend over time
# ---------------------------------------------------------------------
df_sorted = df.copy()
df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
df_sorted = df_sorted.sort_values("Date")
plt.figure(figsize=(10, 5))
plt.plot(df_sorted["Date"], df_sorted["Food_Waste_KG"], alpha=0.4, label="Daily waste")
plt.plot(df_sorted["Date"], df_sorted["Food_Waste_KG"].rolling(14).mean(),
         color="red", linewidth=2, label="14-day rolling average")
plt.title("Food Waste Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Food Waste (kg)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/07_waste_trend.png", dpi=120)
plt.close()

print("\nAll EDA plots saved to plots/ directory.")

print("\n" + "=" * 60)
print("KEY CORRELATIONS WITH FOOD WASTE")
print("=" * 60)
print(df[numeric_cols].corr()["Food_Waste_KG"].sort_values(ascending=False))
