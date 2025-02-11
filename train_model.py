import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("healthh.csv")

# Drop unnecessary columns
df.drop(columns=["Visit ID", "Patient ID", "Hospital ID", "Hospital Name", "Visit Date"], inplace=True, errors='ignore')

# Define categorical and numeric columns
categorical_cols = ["Region", "Day of Week", "Season", "Time of Day", "Urgency Level", "Patient Outcome"]
numeric_features = [col for col in df.columns if col not in categorical_cols and col != "Total Wait Time (min)"]

# Label Encoding for categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    joblib.dump(le, f"{col}_label_encoder.pkl")  # Save label encoders

# Standard Scaling for numerical features
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])
joblib.dump(scaler, "scaler.pkl")  # Save the scaler

# Split data
X = df.drop(columns=["Total Wait Time (min)"])
y = df["Total Wait Time (min)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "xgboost_wait_time.pkl")
