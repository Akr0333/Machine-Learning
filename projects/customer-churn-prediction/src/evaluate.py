"""Evaluate a trained churn pipeline with business-friendly metrics."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(ROOT / "data" / "churn.csv"))
parser.add_argument("--model", default=str(ROOT / "models" / "churn_pipeline.joblib"))
args = parser.parse_args()

df = pd.read_csv(args.data)
y = df.pop("churn").map({"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0})
model = joblib.load(args.model)
pred = model.predict(df)
prob = model.predict_proba(df)[:, 1]

print("=== Customer Churn Model Evaluation ===")
print(f"Accuracy : {accuracy_score(y, pred):.3f}")
print(f"Precision: {precision_score(y, pred, zero_division=0):.3f}")
print(f"Recall   : {recall_score(y, pred, zero_division=0):.3f}")
print(f"F1 Score : {f1_score(y, pred, zero_division=0):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y, prob):.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y, pred))
print("\nClassification Report:\n", classification_report(y, pred, zero_division=0))
