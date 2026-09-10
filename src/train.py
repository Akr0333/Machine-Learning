from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "housing.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "house_price_model.joblib"
TARGET = "price"
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "distance_km"]
CATEGORICAL = ["location", "parking", "furnished"]


def build_pipeline() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    model = RandomForestRegressor(
        n_estimators=350,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train() -> dict[str, float]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.generate_data` first.")

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(mean_squared_error(y_test, predictions) ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("Model trained successfully")
    print(f"MAE : ₹{metrics['MAE']:,.0f}")
    print(f"RMSE: ₹{metrics['RMSE']:,.0f}")
    print(f"R²  : {metrics['R2']:.4f}")
    print(f"Saved model to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train()
