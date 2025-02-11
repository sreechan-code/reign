import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load trained model from .pkl file
model = joblib.load("xgboost_wait_time.pkl")

# Load label encoders
categorical_cols = ["Region", "Day of Week", "Season", "Time of Day", "Urgency Level", "Patient Outcome"]
label_encoders = {col: joblib.load(f"{col}_label_encoder.pkl") for col in categorical_cols}

# Load sample dataset for reference
df_sample = pd.read_csv("healthh.csv")
df_sample.drop(columns=["Visit ID", "Patient ID", "Hospital ID", "Hospital Name", "Visit Date"], inplace=True, errors='ignore')

# Identify numeric features
numeric_features = [col for col in df_sample.columns if col not in categorical_cols and col != "Total Wait Time (min)"]
scaler = joblib.load("scaler.pkl")

# Streamlit App UI
st.set_page_config(page_title="Smart Healthcare Appointment Scheduler", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            color: #2c3e50;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
        }
        .sub-header {
            color: #34495e;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-size: 18px;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #27ae60;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display Logo
st.image("reign clinic.png", width=300)

# App Title
st.markdown('<div class="main-title">Smart Healthcare Appointment Scheduler</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict Patient Wait Time</div>', unsafe_allow_html=True)

# User Inputs
user_inputs = {}
for col in categorical_cols:
    user_inputs[col] = st.selectbox(col, df_sample[col].astype(str).unique())

for col in numeric_features:
    user_inputs[col] = st.number_input(f"{col}", value=float(df_sample[col].mean()))

# Prediction Button
if st.button("Predict Wait Time"):
    # Encode categorical inputs
    encoded_inputs = {}
    for col in categorical_cols:
        if user_inputs[col] in label_encoders[col].classes_:
            encoded_inputs[col] = label_encoders[col].transform([user_inputs[col]])[0]
        else:
            st.warning(f"Warning: '{user_inputs[col]}' is a new category. Using a default value.")
            encoded_inputs[col] = -1  

    # Create DataFrame with correct feature names
    input_data = pd.DataFrame([{**encoded_inputs, **{col: user_inputs[col] for col in numeric_features}}])
    
    # Ensure correct feature order
    input_data = input_data[df_sample.drop(columns=["Total Wait Time (min)"]).columns]
    
    # Standardize numerical features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Make prediction
    predicted_wait_time = model.predict(input_data)[0]
    
    st.success(f"Predicted Wait Time: {predicted_wait_time:.2f} minutes")
