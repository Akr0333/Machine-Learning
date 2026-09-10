"""Train and evaluate a customer churn classifier.

Expected CSV columns: customer_id, tenure, monthly_charges, total_charges,
contract, payment_method, internet_service, senior_citizen, churn.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "churn.csv"
MODEL = ROOT / "models" / "churn_pipeline.joblib"


def main():
    df = pd.read_csv(DATA)
    target = "churn"
    if target not in df.columns:
        raise ValueError("Dataset must contain a 'churn' column")

    X = df.drop(columns=[target, "customer_id"], errors="ignore")
    y = df[target].astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if y.isna().any():
        raise ValueError("Churn values must be Yes/No or 1/0")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, predictions))
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()
