"""Lightweight model explanation utilities.

For the portfolio, this module reports feature importance for tree-based
models when available. For the baseline logistic model, the dashboard can
surface model coefficients after preprocessing.
"""

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline.joblib"

def get_logistic_coefficients():
    pipeline = joblib.load(MODEL)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    return sorted(
        zip(names, model.coef_[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

if __name__ == "__main__":
    for name, coefficient in get_logistic_coefficients()[:15]:
        print(f"{name}: {coefficient:.4f}")
