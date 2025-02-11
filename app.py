import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Debugging: Print current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Define file paths
model_filename = "xgboost_wait_time.pkl"

# Check if the model file exists
if not os.path.exists(model_filename):
    st.error("🚨 Model file not found! Please ensure `xgboost_wait_time.pkl` is uploaded to GitHub and available in Streamlit Cloud.")
    st.stop()

# Load the trained XGBoost model
model = joblib.load(model_filename)
st.success("✅ Model loaded successfully!")

# Load label encoders
categorical_cols = ["Region", "Day of Week", "Season", "Time of Day", "Urgency Level", "Patient Outcome"]
label_encoders = {}
for col in categorical_cols:
    encoder_filename = f"{col}_label_encoder.pkl"
    if os.path.exists(encoder_filename):
        label_encoders[col] = joblib.load(encoder_filename)
    else:
        st.error(f"🚨 Missing file: {encoder_filename}")
        st.stop()

# Load StandardScaler
scaler_filename = "scaler.pkl"
if os.path.exists(scaler_filename):
    scaler = joblib.load(scaler_filename)
else:
    st.error("🚨 Missing file: scaler.pkl")
    st.stop()

st.success("✅ All required files are loaded successfully!")

