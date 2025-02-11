import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Check if model file exists
model_path = "xgboost_wait_time.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found! Please upload `xgboost_wait_time.pkl` to the repository.")
    st.stop()

# Load trained model
model = joblib.load(model_path)

# Load label encoders
categorical_cols = ["Region", "Day of Week", "Season", "Time of Day", "Urgency Level", "Patient Outcome"]
label_encoders = {}
for col in categorical_cols:
    encoder_path = f"{col}_label_encoder.pkl"
    if os.path.exists(encoder_path):
        label_encoders[col] = joblib.load(encoder_path)
    else:
        st.error(f"Missing file: {encoder_path}")
        st.stop()

# Load StandardScaler
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    st.error("Missing file: scaler.pkl")
    st.stop()

scaler = joblib.load(scaler_path)

# Load sample dataset
df_sample_path = "healthh.csv"
if os.path.exists(df_sample_path):
    df_sample = pd.read_csv(df_sample_path)
else:
    st.error("Missing sample dataset: healthh.csv")
    st.stop()

df_sample.drop(columns=["Visit ID", "Patient ID", "Hospital ID", "Hospital Name", "Visit Date"], inplace=True, errors='ignore')
numeric_features = [col for col in df_sample.columns if col not in categorical_cols and col != "Total Wait Time (min)"]

# Streamlit UI
st.set_page_config(page_title="Smart Healthcare Scheduler", layout="centered")
st.image("reign_clinic.png", width=300)

st.title("Smart Healthcare Appointment Scheduler")
st.subheader("Predict Patient Wait Time")

# User Inputs
user_inputs = {}
for col in categorical_cols:
    user_inputs[col] = st.selectbox(col, df_sample[col].astype(str).unique())

for col in numeric_features:
    user_inputs[col] = st.number_input(f"{col}", value=float(df_sample[col].mean()))

# Prediction Button
if st.button("Predict Wait Time"):
    # Encode categorical inputs
    encoded_inputs = {col: label_encoders[col].transform([user_inputs[col]])[0] for col in categorical_cols}

    # Create DataFrame
    input_data = pd.DataFrame([{**encoded_inputs, **{col: user_inputs[col] for col in numeric_features}}])
    
    # Standardize numerical features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Make prediction
    predicted_wait_time = model.predict(input_data)[0]
    
    st.success(f"Predicted Wait Time: {predicted_wait_time:.2f} minutes")
