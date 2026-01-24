import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Energy Production Predictor", page_icon="⚡")

st.title("⚡ Energy Production Predictor")
st.write("Enter inputs and get a predicted energy production output.")

@st.cache_resource
def load_model():
    return joblib.load("tuned_gradient_boosting_pipeline.joblib")

model = load_model()

# ---- UI inputs ----
start_hour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=12)
day_of_year = st.number_input("Day of Year (1-366)", min_value=1, max_value=366, value=100)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)

source = st.selectbox("Source", ["Solar", "Wind", "Hydro", "Coal", "Gas", "Other"])
day_name = st.selectbox("Day Name", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
month_name = st.selectbox("Month Name", ["January","February","March","April","May","June","July","August","September","October","November","December"])
season = st.selectbox("Season", ["Spring","Summer","Autumn","Winter"])

# ---- Feature engineering (must match training) ----
hour_rad = 2 * np.pi * (start_hour / 24)
doy_rad = 2 * np.pi * (day_of_year / 365)

row = {
    "Hour_sin": np.sin(hour_rad),
    "Hour_cos": np.cos(hour_rad),
    "Doy_sin": np.sin(doy_rad),
    "Doy_cos": np.cos(doy_rad),
    "Year": year,
    "Source": source,
    "Day_Name": day_name,
    "Month_Name": month_name,
    "Season": season
}

input_df = pd.DataFrame([row])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Production: {pred:,.2f}")
