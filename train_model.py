"""
train_model.py
---------------
Trains and compares multiple regression models to predict canteen food
waste (kg), evaluates them with MAE / MSE / RMSE / R2, and saves the
best-performing model (plus feature importance) to the model/ directory
for use by app.py.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_raw_data, preprocess, FEATURE_COLUMNS

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load & preprocess
# ---------------------------------------------------------------------
df = load_raw_data()
X, y = preprocess(df, is_training=True)

# ---------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# ---------------------------------------------------------------------
# 3. Train & compare models
# ---------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    print(f"\n{name}")
    print(f"  MAE  : {mae:.3f} kg")
    print(f"  MSE  : {mse:.3f}")
    print(f"  RMSE : {rmse:.3f} kg")
    print(f"  R2   : {r2:.3f}")

# ---------------------------------------------------------------------
# 4. Select the best model (highest R2, tie-break on lowest RMSE)
# ---------------------------------------------------------------------
best_name = max(results, key=lambda n: (results[n]["R2"], -results[n]["RMSE"]))
best_model = models[best_name]
print(f"\n>>> Best model selected: {best_name} (R2={results[best_name]['R2']:.3f}, "
      f"RMSE={results[best_name]['RMSE']:.3f} kg)")

# ---------------------------------------------------------------------
# 5. Save model, metrics, and feature list
# ---------------------------------------------------------------------
joblib.dump(best_model, "model/best_model.pkl")

with open("model/metrics.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

with open("model/feature_columns.json", "w") as f:
    json.dump(FEATURE_COLUMNS, f, indent=2)

print("\nSaved -> model/best_model.pkl, model/metrics.json, model/feature_columns.json")

# ---------------------------------------------------------------------
# 6. Plots: model comparison + feature importance + actual vs predicted
# ---------------------------------------------------------------------
metric_names = ["MAE", "RMSE", "R2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, metric in zip(axes, metric_names):
    values = [results[n][metric] for n in models]
    ax.bar(list(models.keys()), values, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=20)
plt.suptitle("Model Comparison")
plt.tight_layout()
plt.savefig("plots/08_model_comparison.png", dpi=120)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=True)
    plt.figure(figsize=(8, 7))
    importances.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance ({best_name})")
    plt.tight_layout()
    plt.savefig("plots/09_feature_importance.png", dpi=120)
    plt.close()

plt.figure(figsize=(6, 6))
best_preds = predictions[best_name]
plt.scatter(y_test, best_preds, alpha=0.5, color="darkorange")
lims = [0, max(y_test.max(), best_preds.max()) + 2]
plt.plot(lims, lims, "k--", linewidth=1)
plt.xlabel("Actual Food Waste (kg)")
plt.ylabel("Predicted Food Waste (kg)")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout()
plt.savefig("plots/10_actual_vs_predicted.png", dpi=120)
plt.close()

print("Saved comparison/feature-importance/actual-vs-predicted plots to plots/.")
