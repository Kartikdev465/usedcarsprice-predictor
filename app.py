import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and training column structure
model = joblib.load("used_car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App Title
st.title("🚗 Used Car Price Predictor")

st.markdown("### Fill out the details below to get the estimated price:")

# Input fields
brand = st.selectbox("Brand", ['Ford', 'Hyundai', 'Lexus', 'INFINITI', 'Audi', 'Acura', 'BMW', 'Tesla'])
model_name = st.text_input("Model", "Model X Long Range Plus")
model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2020)
milage = st.number_input("Mileage (mi.)", min_value=0, max_value=300000, value=30000)
engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0)
accident = st.selectbox("Any Accidents?", ['No', 'Yes'])
fuel_type = st.selectbox("Fuel Type", ['Gasoline', 'Diesel', 'Electric', 'Hybrid', 'E85 Flex Fuel'])
transmission = st.selectbox("Transmission", ['Automatic', 'Manual', '6-Speed A/T', '8-Speed Automatic', 'A/T'])

# Convert 'Yes'/'No' to 1/0
accident_val = 1 if accident == "Yes" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    'brand': brand,
    'model': model_name,
    'model_year': model_year,
    'milage': milage,
    'engine': engine,
    'accident': accident_val,
    'fuel_type': fuel_type,
    'transmission': transmission
}])

# One-hot encode input
input_encoded = pd.get_dummies(input_data)

# Align with training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Predict button
if st.button("🔮 Predict Price"):
    price = model.predict(input_encoded)[0]
    st.success(f"💰 Estimated Price: **${price:,.2f}**")
