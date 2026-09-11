"""End-to-end loan approval classification example."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = pd.DataFrame([
    [25, 35000, 650, "Yes", 1], [31, 52000, 720, "Yes", 1],
    [45, 28000, 590, "No", 0], [29, 70000, 760, "Yes", 1],
    [52, 40000, 610, "No", 0], [36, 62000, 705, "Yes", 1],
    [41, 45000, 640, "No", 0], [27, 58000, 735, "Yes", 1],
    [49, 33000, 600, "No", 0], [33, 68000, 750, "Yes", 1],
    [38, 50000, 690, "Yes", 1], [55, 30000, 570, "No", 0],
    [24, 42000, 680, "Yes", 1], [47, 39000, 620, "No", 0],
    [30, 61000, 710, "Yes", 1], [43, 36000, 605, "No", 0],
], columns=["age", "income", "credit_score", "employment", "approved"])

X = DATA.drop(columns="approved")
y = DATA["approved"]

categorical = ["employment"]
numeric = ["age", "income", "credit_score"]
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
}

for name, estimator in models.items():
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name}\n")
    print(classification_report(y_test, predictions, zero_division=0))
