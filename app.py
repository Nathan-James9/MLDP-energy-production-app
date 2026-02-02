import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Energy Production Predictor", page_icon="⚡")

st.title("⚡ Energy Production Predictor")
st.write("Pick a date, start hour, and source to predict energy production.")

# ---- Load model -----
@st.cache_resource
def load_model():
    return joblib.load("best_energy_model.pkl")

model = load_model()

# ---- Load choices + derive Month->Season mapping from dataset ----
@st.cache_data
def load_choices_and_maps():
    df = pd.read_csv("Energy Production Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Choices for Source only (we won't ask user for month/day/season anymore)
    sources = sorted(df["Source"].dropna().unique().tolist())

    # Build a month->season mapping based on most common season label in dataset
    # This ensures our Season matches the dataset’s definition.
    month_season_map = (
        df.dropna(subset=["Month_Name", "Season"])
          .groupby("Month_Name")["Season"]
          .agg(lambda s: s.value_counts().index[0])
          .to_dict()
    )

    return sources, month_season_map

sources, month_season_map = load_choices_and_maps()

# ---- UI inputs (simplified) ----
picked_date = st.date_input("Date", value=pd.to_datetime("2024-04-10").date())
start_hour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=12)
source = st.selectbox("Source", sources)

# ---- Derive calendar fields from Date ----
dt = pd.to_datetime(picked_date)

year = int(dt.year)
day_of_year = int(dt.dayofyear)
day_name = dt.strftime("%A")       # e.g., "Monday"
month_name = dt.strftime("%B")     # e.g., "January"

# Use dataset-derived season mapping; fallback if missing
season = month_season_map.get(month_name, "Unknown")

if season == "Unknown":
    st.warning("Season label could not be derived for this month based on the dataset.")


# ---- Feature engineering (must match training) ----
hour_rad = 2 * np.pi * (start_hour / 24)
doy_rad = 2 * np.pi * (day_of_year / 365)  # keep consistent with training choice

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

# Optional: show derived fields (nice for debugging + lecturer demo)
with st.expander("See derived inputs (for transparency)"):
    st.write({
        "Year": year,
        "Day_of_Year": day_of_year,
        "Day_Name": day_name,
        "Month_Name": month_name,
        "Season (derived from dataset)": season
    })
    st.dataframe(input_df)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Production: {pred:,.2f}")
