from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Energy Production Predictor",
    page_icon="⚡",
    layout="centered"
)

# -----------------------------
# Light custom styling
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0f1117;
}
.card {
    background-color: #1a1d24;
    padding: 1.5rem;
    border-radius: 14px;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.05);
}
.big-result {
    font-size: 2rem;
    font-weight: 700;
    color: #00e0a4;
}
.subtle {
    color: #9aa0aa;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("## ⚡ Energy Production Predictor")
st.markdown(
    "<p class='subtle'>Select a source and a date/time to estimate energy production output.</p>",
    unsafe_allow_html=True
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_energy_model.pkl")

model = load_model()

# -----------------------------
# Load lookup table
# -----------------------------
@st.cache_data
def load_lookup():
    lk = pd.read_parquet("feature_lookup.parquet")
    lk["Date"] = pd.to_datetime(lk["Date"]).dt.date
    return lk

lookup = load_lookup()
sources = sorted(lookup["Source"].dropna().unique().tolist())
HAS_PROD = "Production" in lookup.columns

# -----------------------------
# Helper functions (unchanged)
# -----------------------------
def cyclical_hour_features(h):
    rad = 2 * np.pi * (h / 24)
    return np.sin(rad), np.cos(rad)

def cyclical_doy_features(doy):
    rad = 2 * np.pi * (doy / 365)
    return np.sin(rad), np.cos(rad)

def build_typical_lag_estimates_from_lookup(lk, source, month_num, start_hour):
    temp = lk.copy()
    temp["Month_Num_from_Date"] = pd.to_datetime(temp["Date"]).dt.month

    sub = temp[(temp["Source"] == source) & (temp["Month_Num_from_Date"] == month_num)]
    if sub.empty:
        sub = temp[temp["Source"] == source]

    same_hour = sub[sub["Start_Hour"] == start_hour]["Production"]
    typical_now = float(same_hour.median()) if not same_hour.empty else float(sub["Production"].median())

    prev_hour = (start_hour - 1) % 24
    prev_vals = sub[sub["Start_Hour"] == prev_hour]["Production"]
    typical_lag1 = float(prev_vals.median()) if not prev_vals.empty else typical_now

    typical_lag24 = typical_now
    typical_roll3 = float(np.mean([typical_now, typical_lag1, typical_lag24]))
    typical_roll24 = float(sub["Production"].median())
    typical_roll24_std = float(sub["Production"].std(ddof=0)) if len(sub) > 1 else 0.0

    return {
        "Prod_lag1": typical_lag1,
        "Prod_lag24": typical_lag24,
        "Prod_roll3": typical_roll3,
        "Prod_roll24": typical_roll24,
        "Prod_roll24_std": typical_roll24_std
    }

def make_input_row(user_source, user_date, user_hour):
    doy = pd.Timestamp(user_date).dayofyear
    year = user_date.year
    is_weekend = 1 if pd.Timestamp(user_date).dayofweek >= 5 else 0

    hour_sin, hour_cos = cyclical_hour_features(user_hour)
    doy_sin, doy_cos = cyclical_doy_features(doy)

    match = lookup[
        (lookup["Source"] == user_source) &
        (lookup["Date"] == user_date) &
        (lookup["Start_Hour"] == user_hour)
    ]

    if len(match) > 0:
        row = match.iloc[0]
        lag_feats = {
            "Prod_lag1": float(row["Prod_lag1"]),
            "Prod_lag24": float(row["Prod_lag24"]),
            "Prod_roll3": float(row["Prod_roll3"]),
            "Prod_roll24": float(row["Prod_roll24"]),
            "Prod_roll24_std": float(row.get("Prod_roll24_std", 0.0))
        }
        used_mode = "Historical prediction (Exact lag features from data)"
    else:
        if not HAS_PROD:
            raise ValueError("Planning mode needs Production in feature_lookup.parquet.")
        lag_feats = build_typical_lag_estimates_from_lookup(
            lookup, user_source, user_date.month, user_hour
        )
        used_mode = "Planning estimate (based on typical historical patterns)"

    input_dict = {
        "Source": user_source,
        "Year": year,
        "Is_Weekend": is_weekend,
        "Hour_sin": hour_sin,
        "Hour_cos": hour_cos,
        "Doy_sin": doy_sin,
        "Doy_cos": doy_cos,
        **lag_feats
    }

    return pd.DataFrame([input_dict]), used_mode

# -----------------------------
# Input card
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Energy Source", sources)

with col2:
    start_hour = st.number_input("Start Hour (0–23)", 0, 23, 12)

user_date = st.date_input("Date", value=date(2025, 1, 1))

min_date = lookup["Date"].min()
max_date = lookup["Date"].max()

st.markdown(
    f"<small style='color:#9aa0a6;'>Dataset coverage (historical mode): {min_date} to {max_date}</small>",
    unsafe_allow_html=True
)
st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1.5rem;'>", unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("⚡ Predict Energy Output", use_container_width=True):
    try:
        input_df, mode_used = make_input_row(source, user_date, start_hour)
        pred = model.predict(input_df)[0]

        st.markdown(
            f"<div class='big-result'>{pred:,.2f}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Mode used: {mode_used}")

        if "Planning" in mode_used:
            st.info("Estimate based on typical historical patterns — ideal for future planning.")

    except Exception as e:
        st.error("Prediction failed.")
        st.write(str(e))
        
