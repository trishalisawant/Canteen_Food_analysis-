"""
app.py
------
Streamlit web application: "Food Waste Prediction in College Canteens"

Run with:
    streamlit run app.py

The app loads the pre-trained model (model/best_model.pkl). If the model
file is missing, it trains one on the fly from data/canteen_data.csv so
the app never breaks on first run.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st

from preprocessing import build_single_input, DAY_ORDER, MEAL_TYPES

st.set_page_config(page_title="Canteen Food Waste Predictor", page_icon="🍽️", layout="centered")

MODEL_PATH = "model/best_model.pkl"
METRICS_PATH = "model/metrics.json"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Fallback: train a quick model on the spot if none is saved yet.
        import subprocess
        subprocess.run(["python3", "train_model.py"], check=True)
    model = joblib.load(MODEL_PATH)
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return model, metrics


model, metrics = load_model()

st.title("🍽️ College Canteen Food Waste Predictor")
st.write(
    "Estimate how much food will be wasted for a given day so the canteen "
    "can prepare the right quantity of food and cut down on waste."
)

if metrics:
    best_name = metrics["best_model"]
    best_r2 = metrics["results"][best_name]["R2"]
    best_rmse = metrics["results"][best_name]["RMSE"]
    st.caption(
        f"Model in use: **{best_name}**  |  R² = {best_r2:.2f}  |  "
        f"Typical error (RMSE) ≈ {best_rmse:.2f} kg"
    )

st.divider()
st.subheader("Enter Today's Canteen Details")

col1, col2 = st.columns(2)

with col1:
    students_expected = st.number_input(
        "Number of students expected", min_value=0, max_value=5000, value=650, step=10
    )
    meals_prepared = st.number_input(
        "Meals to be prepared", min_value=0, max_value=5000, value=620, step=10
    )
    meals_served = st.number_input(
        "Expected meals served", min_value=0, max_value=5000, value=590, step=10
    )
    previous_waste = st.number_input(
        "Previous day's food waste (kg)", min_value=0.0, max_value=200.0, value=12.0, step=0.5
    )
    avg_waste_7d = st.number_input(
        "Average food waste over last 7 days (kg)", min_value=0.0, max_value=200.0,
        value=10.0, step=0.5,
        help="If unsure, use roughly the same value as previous day's waste."
    )

with col2:
    day = st.selectbox("Day of the week", DAY_ORDER, index=0)
    holiday = st.radio("Is it a holiday?", ["No", "Yes"], horizontal=True)
    special_event = st.radio("Special event / college function?", ["No", "Yes"], horizontal=True)
    meal_type = st.selectbox("Meal type", MEAL_TYPES, index=1)

if meals_served > meals_prepared:
    st.warning("Meals served cannot exceed meals prepared — please check your inputs.")

st.divider()

if st.button("🔮 Predict Food Waste", type="primary", use_container_width=True):
    X_input = build_single_input(
        students_expected=students_expected,
        meals_prepared=meals_prepared,
        meals_served=meals_served,
        previous_waste=previous_waste,
        avg_waste_7d=avg_waste_7d,
        day=day,
        holiday=holiday,
        special_event=special_event,
        meal_type=meal_type,
    )

    prediction = max(0.0, float(model.predict(X_input)[0]))

    st.success(f"### Predicted Food Waste: **{prediction:.1f} kg**")

    # Simple guidance based on the prediction
    if prediction < 5:
        st.info("Waste level looks low — current preparation quantity seems well matched to demand.")
    elif prediction < 15:
        st.info("Moderate waste expected — consider trimming preparation quantity slightly.")
    else:
        st.warning(
            "High waste expected — consider reducing the quantity prepared or reviewing "
            "portion sizes for this day."
        )

    with st.expander("See the exact features sent to the model"):
        st.dataframe(X_input.T.rename(columns={0: "value"}))

st.divider()
with st.expander("ℹ️ About this app"):
    st.write(
        "This tool is trained on historical canteen data (students expected, meals "
        "prepared/served, previous waste trends, day of week, holidays, special events, "
        "and meal type) using a regression model (Linear Regression / Decision Tree / "
        "Random Forest — the best performer on held-out data is used automatically). "
        "Predictions are estimates to support planning, not exact guarantees."
    )
