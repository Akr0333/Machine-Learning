import streamlit as st

from src.predict import predict_price

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.caption("End-to-end machine learning demo using a Random Forest regression pipeline.")

with st.form("prediction_form"):
    area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    age = st.slider("Property age (years)", 0, 50, 8)
    distance = st.number_input("Distance from city centre (km)", min_value=0.5, max_value=80.0, value=8.0, step=0.5)
    location = st.selectbox("Location", ["Central", "Suburban", "Outskirts"])
    parking = st.selectbox("Parking", ["Yes", "No"])
    furnished = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    submitted = st.form_submit_button("Predict price")

if submitted:
    features = {
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "distance_km": distance,
        "location": location,
        "parking": parking,
        "furnished": furnished,
    }
    try:
        prediction = predict_price(features)
        st.success(f"Estimated price: ₹{prediction:,.0f}")
    except FileNotFoundError:
        st.error("Model not found. Generate the data and train the model first.")
