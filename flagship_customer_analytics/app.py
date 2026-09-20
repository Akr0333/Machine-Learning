"""Streamlit dashboard for Customer Intelligence & Churn Analytics."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "customers.csv"
st.set_page_config(page_title="Customer Intelligence", page_icon="📈", layout="wide")

st.title("📈 Customer Intelligence Dashboard")
st.caption("Analytics + machine learning decision support for customer retention")

if not DATA.exists():
    st.warning("Demo data not found. Run: python src/generate_data.py")
    st.stop()

df = pd.read_csv(DATA)
customers = len(df)
churn_rate = df["churn"].mean() * 100
revenue_at_risk = df.loc[df["churn"] == 1, "monthly_revenue"].sum()
avg_revenue = df["monthly_revenue"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", "%d" % customers)
c2.metric("Churn Rate", "%.1f%%" % churn_rate)
c3.metric("Revenue at Risk", "$%,.0f" % revenue_at_risk)
c4.metric("Avg Monthly Revenue", "$%,.0f" % avg_revenue)

st.subheader("Churn by Contract")
contract = df.groupby("contract_type")["churn"].mean().mul(100).sort_values(ascending=False)
st.bar_chart(contract)

st.subheader("Churn by Segment")
segment = df.groupby("segment")["churn"].mean().mul(100).sort_values(ascending=False)
st.bar_chart(segment)

st.subheader("Revenue at Risk by Segment")
risk_revenue = df[df["churn"] == 1].groupby("segment")["monthly_revenue"].sum().sort_values(ascending=False)
st.bar_chart(risk_revenue)

st.subheader("High-Value Customer Risk")
threshold = st.slider(
    "Minimum monthly revenue",
    min_value=float(df["monthly_revenue"].min()),
    max_value=float(df["monthly_revenue"].quantile(0.95)),
    value=float(df["monthly_revenue"].median()),
)
risk = df[(df["churn"] == 1) & (df["monthly_revenue"] >= threshold)].sort_values(
    "monthly_revenue", ascending=False
)
st.dataframe(
    risk[["customer_id", "segment", "monthly_revenue", "tenure_months",
          "support_tickets", "contract_type"]].head(50),
    use_container_width=True,
)

st.info("Demo data is synthetic. Replace it with an approved business dataset before using this for operational decisions.")
