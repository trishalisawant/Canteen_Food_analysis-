"""
generate_dataset.py
--------------------
Generates a realistic synthetic historical dataset for a college canteen.

Since real canteen records are usually not publicly available, this script
builds a dataset that follows realistic relationships between the features
and food waste, so the downstream ML pipeline has genuine patterns to learn
(rather than pure noise). If you have real canteen records, simply replace
data/canteen_data.csv with your own file that follows the same column
schema and skip this script.

Columns produced:
    Date, Day, Students_Expected, Meals_Prepared, Meals_Served,
    Previous_Waste_KG, Avg_Waste_Last_7Days, Holiday, Special_Event,
    Meal_Type, Food_Waste_KG
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_DAYS = 730          # 2 years of daily records
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]
DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]

start_date = datetime(2023, 6, 1)

records = []
prev_waste = 8.0                  # seed value for the recursive "previous day's waste"
waste_history = []                # rolling history to compute a 7-day average

for i in range(N_DAYS):
    date = start_date + timedelta(days=i)
    day_name = DAYS_OF_WEEK[date.weekday()]
    is_weekend = day_name in ["Saturday", "Sunday"]

    # ---- Holiday & special events (roughly 8% holiday, 6% special event) ----
    holiday = np.random.choice(["Yes", "No"], p=[0.08, 0.92])
    special_event = np.random.choice(["Yes", "No"], p=[0.06, 0.94])

    # ---- Meal type (each day has one record per meal in real deployments;
    #      here we sample a meal type per record to keep the dataset general) ----
    meal_type = np.random.choice(MEAL_TYPES, p=[0.30, 0.45, 0.25])

    # ---- Base attendance model ----
    base_students = 700
    if is_weekend:
        base_students -= 250          # far fewer students on campus
    if holiday == "Yes":
        base_students -= 400          # holidays sharply cut attendance
    if special_event == "Yes":
        base_students += 150          # events draw extra footfall
    if meal_type == "Breakfast":
        base_students *= 0.55
    elif meal_type == "Dinner":
        base_students *= 0.75

    students_expected = max(
        30, int(np.random.normal(base_students, 45))
    )

    # ---- Meals prepared: canteen staff usually over-prepare by a margin,
    #      influenced by expected students and a planning buffer ----
    planning_buffer = np.random.uniform(1.03, 1.15)
    meals_prepared = int(students_expected * planning_buffer)

    # ---- Meals served: close to students expected but with turnout noise ----
    turnout_rate = np.random.normal(0.93, 0.05)
    turnout_rate = min(max(turnout_rate, 0.6), 1.0)
    meals_served = int(min(meals_prepared, students_expected * turnout_rate))

    # ---- Food waste model ----
    # Waste grows with the gap between prepared and served meals, and has
    # some autocorrelation with previous day's waste (kitchen habits persist).
    unused_meals = max(0, meals_prepared - meals_served)
    waste_per_unused_meal = np.random.uniform(0.14, 0.22)   # kg per unused meal
    base_waste = unused_meals * waste_per_unused_meal

    autocorrelation = 0.25 * prev_waste
    holiday_extra = 3.0 if holiday == "Yes" else 0.0
    event_extra = 2.0 if special_event == "Yes" else 0.0
    weekend_extra = 1.5 if is_weekend else 0.0
    noise = np.random.normal(0, 1.5)

    food_waste = max(
        0.5,
        base_waste * 0.6 + autocorrelation + holiday_extra * 0.3
        + event_extra * 0.3 + weekend_extra * 0.3 + noise
    )
    food_waste = round(food_waste, 2)

    waste_history.append(food_waste)
    avg_waste_7d = round(np.mean(waste_history[-7:]), 2)

    records.append({
        "Date": date.strftime("%Y-%m-%d"),
        "Day": day_name,
        "Students_Expected": students_expected,
        "Meals_Prepared": meals_prepared,
        "Meals_Served": meals_served,
        "Previous_Waste_KG": round(prev_waste, 2),
        "Avg_Waste_Last_7Days": avg_waste_7d,
        "Holiday": holiday,
        "Special_Event": special_event,
        "Meal_Type": meal_type,
        "Food_Waste_KG": food_waste,
    })

    prev_waste = food_waste

df = pd.DataFrame(records)
df.to_csv("data/canteen_data.csv", index=False)
print(f"Dataset generated: {len(df)} records -> data/canteen_data.csv")
print(df.head())
