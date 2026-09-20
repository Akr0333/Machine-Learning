"""Train a reproducible customer churn model."""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customers.csv"
MODEL = ROOT / "models" / "churn_pipeline.joblib"

df = pd.read_csv(DATA)

target = "churn"
X = df.drop(columns=[target, "customer_id"])
y = df[target]

categorical = X.select_dtypes(include=["object"]).columns
numeric = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

probabilities = pipeline.predict_proba(X_test)[:, 1]
predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")

MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL)
print(f"Saved model to {MODEL}")
