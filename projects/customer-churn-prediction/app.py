"""Streamlit demo for Customer Churn Prediction."""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Interactive ML demo — enter customer attributes to estimate churn risk.")

if not MODEL_PATH.exists():
    st.error("Model not found. Train the model first with: python src/train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.form("customer_form"):
    customer_id = st.text_input("Customer ID", "DEMO-001")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = pd.DataFrame([{
        "customer_id": customer_id,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "senior_citizen": senior_citizen,
    }])
    probability = float(model.predict_proba(row)[:, 1][0])
    prediction = int(probability >= 0.5)
    st.metric("Churn probability", f"{probability:.1%}")
    if prediction:
        st.error("High churn risk — consider a retention offer.")
    else:
        st.success("Lower churn risk.")
