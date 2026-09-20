from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customers.csv"

def test_dataset_exists():
    assert DATA.exists()

def test_dataset_schema():
    df = pd.read_csv(DATA)
    required = {
        "customer_id", "segment", "tenure_months",
        "monthly_revenue", "support_tickets", "usage_hours",
        "contract_type", "payment_method", "churn"
    }
    assert required.issubset(df.columns)

def test_churn_is_binary():
    df = pd.read_csv(DATA)
    assert set(df["churn"].unique()).issubset({0, 1})

def test_revenue_is_positive():
    df = pd.read_csv(DATA)
    assert (df["monthly_revenue"] > 0).all()
