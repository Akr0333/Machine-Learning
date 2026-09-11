import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Small reproducible demo dataset.
data = pd.DataFrame({
    "amount": [120, 4500, 80, 9200, 250, 7600, 140, 6100, 330, 8400, 190, 5300],
    "hour": [10, 2, 14, 3, 18, 1, 11, 4, 16, 2, 13, 5],
    "merchant": ["grocery", "electronics", "grocery", "travel", "fuel", "electronics", "grocery", "travel", "fuel", "travel", "grocery", "electronics"],
    "country": ["IN", "US", "IN", "UK", "IN", "US", "IN", "UK", "IN", "US", "IN", "US"],
    "fraud": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
})

X = data.drop(columns="fraud")
y = data["fraud"]

cat = ["merchant", "country"]
num = ["amount", "hour"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
}

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    print(f"\n{name}\n")
    print(classification_report(y_test, predictions, zero_division=0))
