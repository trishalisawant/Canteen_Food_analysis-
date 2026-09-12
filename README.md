# 🍽️ Food Waste Prediction in College Canteens

A machine learning system that predicts the amount of food waste (in kg)
expected in a college canteen on a given day, so kitchen staff can prepare
an appropriate quantity of food and cut down on unnecessary waste.

---

## 1. Problem Statement

Canteens routinely over- or under-prepare food because staff rely on
intuition rather than data. This project predicts expected food waste
from real-world planning factors — expected footfall, meals prepared,
day of week, holidays, special events, and recent waste trends — using a
regression model, and serves the prediction through a simple web app.

---

## 2. Project Structure

```
food_waste_prediction/
├── data/
│   └── canteen_data.csv          # Historical dataset (730 daily records)
├── model/
│   ├── best_model.pkl            # Trained, saved regression model
│   ├── metrics.json              # Evaluation metrics for all models tried
│   └── feature_columns.json      # Exact feature order used by the model
├── plots/                        # EDA & evaluation charts (generated)
├── generate_dataset.py           # Builds the realistic synthetic dataset
├── preprocessing.py              # Shared cleaning / feature engineering / encoding
├── eda.py                        # Exploratory Data Analysis
├── train_model.py                # Trains, compares, evaluates, saves best model
├── app.py                        # Streamlit web application
└── README.md                     # This file
```

---

## 3. Dataset

**File:** `data/canteen_data.csv` (730 daily records, ~2 years)

| Column | Description |
|---|---|
| `Date` | Calendar date of the record |
| `Day` | Day of the week |
| `Students_Expected` | Number of students expected that meal/day |
| `Meals_Prepared` | Number of meals the kitchen prepared |
| `Meals_Served` | Number of meals actually served |
| `Previous_Waste_KG` | Food waste (kg) recorded the previous day |
| `Avg_Waste_Last_7Days` | Rolling 7-day average food waste (kg) |
| `Holiday` | Yes/No — whether it was a holiday |
| `Special_Event` | Yes/No — special college event/function |
| `Meal_Type` | Breakfast / Lunch / Dinner |
| `Food_Waste_KG` | **Target** — actual food waste recorded (kg) |

This dataset is **synthetically generated** (`generate_dataset.py`) with
realistic, non-random relationships (e.g. waste scales with the gap
between meals prepared and served, rises on holidays/events, and shows
day-to-day autocorrelation) so the ML pipeline has genuine signal to
learn. **To use real data**, replace `data/canteen_data.csv` with your
own file using the same column names — everything downstream (EDA,
training, app) works unchanged.

---

## 4. Workflow

```
Data Collection → Data Cleaning → EDA → Feature Selection →
Preprocessing → Train/Test Split → Model Training → Model Evaluation →
Prediction → Streamlit UI
```

### Step-by-step

1. **Generate/collect data** — `python3 generate_dataset.py`
2. **Preprocess** — `preprocessing.py` handles deduplication, missing
   values, clipping invalid values, and one-hot encoding of `Day`,
   `Holiday`, `Special_Event`, `Meal_Type`. It also engineers two useful
   features: `Prepared_Served_Gap` (unused meals) and `Is_Weekend`.
3. **EDA** — `python3 eda.py` prints summary statistics and saves 7
   charts to `plots/`: waste distribution, waste by day/holiday/meal
   type, unused-meals vs waste scatter, correlation heatmap, and the
   waste trend over time.
4. **Train & evaluate** — `python3 train_model.py` trains **Linear
   Regression**, **Decision Tree Regressor**, and **Random Forest
   Regressor**, evaluates each with MAE / MSE / RMSE / R², and
   automatically saves the best-performing model to
   `model/best_model.pkl`.
5. **Predict & serve** — `streamlit run app.py` launches the web UI.

---

## 5. Model Evaluation (this run)

| Model | MAE (kg) | RMSE (kg) | R² |
|---|---|---|---|
| Linear Regression | 1.46 | 1.92 | **0.86** |
| Random Forest | 1.63 | 2.13 | 0.83 |
| Decision Tree | 2.05 | 2.72 | 0.72 |

> Exact numbers are saved in `model/metrics.json` and will vary slightly
> if you regenerate the dataset or change the random seed. `train_model.py`
> automatically picks whichever model scores best (highest R², then
> lowest RMSE) — it isn't hardcoded to any one algorithm.

Key correlations with food waste (from EDA): the **gap between meals
prepared and meals served** is by far the strongest driver, followed by
meals prepared and students expected — confirming that *over-preparation*
is the primary cause of waste in this dataset, with holidays/weekends and
recent waste trends as secondary factors.

---

## 6. Running the Project

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

# 2. Generate the dataset (skip if you already have data/canteen_data.csv)
python3 generate_dataset.py

# 3. Run EDA (optional, produces plots/)
python3 eda.py

# 4. Train the model
python3 train_model.py

# 5. Launch the web app
streamlit run app.py
```

The app opens in your browser. Enter the day's details and click
**Predict Food Waste** to get an estimate in kilograms, along with a
simple recommendation (low / moderate / high waste expected).

---

## 7. Example

**Input**
- Students expected: 650
- Meals prepared: 620
- Meals served: 590
- Previous food waste: 12 kg
- Day: Monday · Holiday: No

**Output**
```
Predicted Food Waste: ~9–10 kg
```

---

## 8. Notes & Possible Extensions

- Swap in real canteen records for a production-grade model — the
  synthetic data is a stand-in that preserves realistic relationships.
- Add more granular features if available: menu item, weather, exam
  period, hostel occupancy.
- Try gradient boosting models (e.g. XGBoost/LightGBM) for comparison.
- Persist predictions vs. actuals over time to continuously retrain and
  improve the model (a feedback loop).
