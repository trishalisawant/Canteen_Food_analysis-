"""
preprocessing.py
-----------------
Shared data-cleaning and feature-encoding logic used by both the training
script (train_model.py) and the Streamlit app (app.py), so predictions are
always built the exact same way the model was trained.
"""

import pandas as pd

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner"]

FEATURE_COLUMNS = [
    "Students_Expected", "Meals_Prepared", "Meals_Served",
    "Previous_Waste_KG", "Avg_Waste_Last_7Days",
    "Prepared_Served_Gap", "Is_Weekend",
    "Day_Monday", "Day_Tuesday", "Day_Wednesday", "Day_Thursday",
    "Day_Friday", "Day_Saturday", "Day_Sunday",
    "Holiday_Yes", "Special_Event_Yes",
    "Meal_Type_Breakfast", "Meal_Type_Lunch", "Meal_Type_Dinner",
]

TARGET_COLUMN = "Food_Waste_KG"


def load_raw_data(path="data/canteen_data.csv"):
    """Load the raw CSV dataset."""
    return pd.read_csv(path)


def clean_data(df):
    """
    Basic data-cleaning steps:
    - drop exact duplicate rows
    - drop rows missing critical fields
    - clip impossible/negative values
    - ensure Meals_Served never exceeds Meals_Prepared
    """
    df = df.copy()
    df = df.drop_duplicates()

    critical_cols = ["Students_Expected", "Meals_Prepared", "Meals_Served"]
    df = df.dropna(subset=critical_cols)

    for col in ["Students_Expected", "Meals_Prepared", "Meals_Served",
                "Previous_Waste_KG", "Avg_Waste_Last_7Days", "Food_Waste_KG"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df["Meals_Served"] = df[["Meals_Served", "Meals_Prepared"]].min(axis=1)

    return df.reset_index(drop=True)


def engineer_features(df):
    """Add derived features that help the model."""
    df = df.copy()
    df["Prepared_Served_Gap"] = df["Meals_Prepared"] - df["Meals_Served"]
    df["Is_Weekend"] = df["Day"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def encode_features(df):
    """
    One-hot encode categorical columns (Day, Holiday, Special_Event,
    Meal_Type) using a fixed, known set of categories so the encoding is
    identical between training and inference, even for a single-row input.
    """
    df = df.copy()

    for day in DAY_ORDER:
        df[f"Day_{day}"] = (df["Day"] == day).astype(int)

    df["Holiday_Yes"] = (df["Holiday"] == "Yes").astype(int)
    df["Special_Event_Yes"] = (df["Special_Event"] == "Yes").astype(int)

    for meal in MEAL_TYPES:
        df[f"Meal_Type_{meal}"] = (df["Meal_Type"] == meal).astype(int)

    return df


def preprocess(df, is_training=True):
    """
    Full preprocessing pipeline: clean -> engineer features -> encode.
    Returns (X, y) if is_training=True (y=None if target column absent),
    otherwise returns X only, aligned to FEATURE_COLUMNS.
    """
    df = clean_data(df) if is_training else df.copy()
    df = engineer_features(df)
    df = encode_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    return (X, y) if is_training else X


def build_single_input(students_expected, meals_prepared, meals_served,
                        previous_waste, avg_waste_7d, day, holiday,
                        special_event, meal_type):
    """
    Helper used by the Streamlit app: build a one-row DataFrame from raw
    user inputs and run it through the same preprocessing pipeline.
    """
    row = pd.DataFrame([{
        "Students_Expected": students_expected,
        "Meals_Prepared": meals_prepared,
        "Meals_Served": meals_served,
        "Previous_Waste_KG": previous_waste,
        "Avg_Waste_Last_7Days": avg_waste_7d,
        "Day": day,
        "Holiday": holiday,
        "Special_Event": special_event,
        "Meal_Type": meal_type,
    }])
    return preprocess(row, is_training=False)
