import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model = joblib.load("gradient_boosting_model.pkl")
feature_names = joblib.load("model_features.pkl")

# App title
st.title("Vehicle Maintenance Priority Predictor")

# Input form
st.header("Enter Maintenance Request Details")
with st.form("predict_form"):
    Vehicle = st.number_input("Vehicle Code (encoded)", value=1000000)
    Odometer = st.number_input("Odometer Reading", value=50000)
    maintenance_category = st.selectbox("Maintenance Category",
                                        ['brake_system', 'cleaning', 'engine', 'mechanical', 'service', 'tire'])
    request_day_of_week = st.slider("Request Day of Week", 0, 6, 2)
    request_hour = st.slider("Request Hour", 0, 23, 10)
    request_month = st.slider("Request Month", 1, 12, 1)

    submitted = st.form_submit_button("Predict")

# Encode maintenance_category manually (as in your training)
category_map = {'brake_system': 0, 'cleaning': 1, 'engine': 2, 'mechanical': 3, 'service': 4, 'tire': 5}

# Collect values in a dictionary with all expected features
input_dict = {
    'Vehicle_encoded': Vehicle,
    'Odometer': Odometer,
    'maintenance_category': category_map[maintenance_category],
    'request_day_of_week': request_day_of_week,
    'request_hour': request_hour,
    'request_month': request_month,
    # Fill missing features with default values (e.g., 0)
}

# Ensure all required features are present
for feature in feature_names:
    if feature not in input_dict:
        input_dict[feature] = 0  # Default fallback

# Create input DataFrame with correct order
input_data = pd.DataFrame([[input_dict[feat] for feat in feature_names]], columns=feature_names)

# Predict
prediction = model.predict(input_data)[0]
