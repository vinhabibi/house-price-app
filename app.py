import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load model and preprocessors
@st.cache_resource
def load_models():
    model   = joblib.load('xgb_house_model.pkl')
    imputer = joblib.load('imputer.pkl')
    scaler  = joblib.load('scaler.pkl')
    return model, imputer, scaler

model, imputer, scaler = load_models()

# Header
st.title('🏠 California House Price Predictor')
st.write('Fill in the details below to predict the median house value.')
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader('📍 Location')
    longitude = st.slider('Longitude', -124.0, -114.0, -119.0, step=0.1)
    latitude  = st.slider('Latitude',   32.0,   42.0,   36.0, step=0.1)
    
    st.subheader('🏘️ Neighborhood')
    housing_median_age = st.slider('Housing Median Age', 1, 51, 20)
    median_income      = st.slider('Median Income ($10k)', 0.5, 15.0, 3.0, step=0.1)

with col2:
    st.subheader('🏗️ Housing Stats')
    households               = st.number_input('Households',               min_value=1,   value=500)
    rooms_per_household      = st.number_input('Rooms per Household',      min_value=0.1, value=5.0,  step=0.1)
    bedrooms_per_room        = st.number_input('Bedrooms per Room',        min_value=0.1, value=0.2,  step=0.01)
    population_per_household = st.number_input('Population per Household', min_value=0.1, value=3.0,  step=0.1)

st.divider()

# Predict button
if st.button('🔮 Predict House Value', use_container_width=True):

    # --- FIX: use a DataFrame with column names matching training ---
    input_data = pd.DataFrame([[
        longitude, latitude, housing_median_age,
        households, median_income,
        rooms_per_household, bedrooms_per_room, population_per_household
    ]], columns=[
        'longitude', 'latitude', 'housing_median_age',
        'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
    ])

    input_imputed = imputer.transform(input_data)
    input_scaled  = scaler.transform(input_imputed)

    log_pred = model.predict(input_scaled)
    price    = np.expm1(log_pred)[0]

    st.success(f'### Predicted Median House Value: **${price:,.2f}**')

    col1, col2, col3 = st.columns(3)
    col1.metric('Prediction', f'${price:,.0f}')
    col2.metric('Model', 'XGBoost')
    col3.metric('R² Score', '0.7488')

st.caption('Built with XGBoost · California Housing Dataset · Vincent Kimutai')