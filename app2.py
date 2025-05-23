import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("gradient_boosting_model.pkl")

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

if submitted:
    input_data = pd.DataFrame([{
        'Vehicle_encoded': Vehicle,
        'Odometer': Odometer,
        'maintenance_category': category_map[maintenance_category],
        'request_day_of_week': request_day_of_week,
        'request_hour': request_hour,
        'request_month': request_month,
        # Add more features as required based on your model
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Maintenance Priority: **{prediction}**")
