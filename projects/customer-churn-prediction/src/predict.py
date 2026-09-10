"""Load a trained pipeline and predict churn for a customer CSV."""
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline.joblib"


def main(path: str):
    model = joblib.load(MODEL)
    df = pd.read_csv(path)
    ids = df["customer_id"] if "customer_id" in df else pd.Series(range(len(df)))
    X = df.drop(columns=["customer_id", "churn"], errors="ignore")
    probability = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({"customer_id": ids, "churn_probability": probability.round(4), "prediction": (probability >= 0.5).astype(int)})
    print(result.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/customers.csv")
    main(sys.argv[1])
