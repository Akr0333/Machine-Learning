from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "house_price_model.joblib"


def predict_price(features: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.generate_data` and `python -m src.train` first.")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([features])
    return float(model.predict(row)[0])
