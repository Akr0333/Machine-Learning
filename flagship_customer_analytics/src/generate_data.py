"""Generate a reproducible demo customer dataset for the project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "customers.csv"
rng = np.random.default_rng(42)
n = 1200
df = pd.DataFrame({
    "customer_id": ["C%05d" % i for i in range(1, n + 1)],
    "segment": rng.choice(["Consumer", "SMB", "Enterprise"], n, p=[0.60, 0.30, 0.10]),
    "tenure_months": rng.integers(1, 73, n),
    "monthly_revenue": np.round(rng.lognormal(4.2, 0.55, n), 2),
    "support_tickets": rng.poisson(2.0, n),
    "usage_hours": np.round(rng.gamma(5, 4, n), 1),
    "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
    "payment_method": rng.choice(["Card", "Bank transfer", "UPI"], n),
})
risk = (1.1 * (df["contract_type"] == "Month-to-month").astype(float)
        - 0.018 * df["tenure_months"] + 0.08 * df["support_tickets"]
        - 0.025 * df["usage_hours"])
prob = 1 / (1 + np.exp(-(risk - risk.mean())))
df["churn"] = rng.binomial(1, np.clip(prob, 0.05, 0.85))
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print("Generated %d customers at %s" % (len(df), OUT))
