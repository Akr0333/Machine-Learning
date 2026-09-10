"""Extract interpretable feature names and coefficients from the churn pipeline."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
model = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
weights = classifier.coef_[0]
result = pd.DataFrame({"feature": feature_names, "coefficient": weights})
result["abs_coefficient"] = result["coefficient"].abs()
result = result.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")

out = ROOT / "reports" / "feature_importance.csv"
result.to_csv(out, index=False)
print(result.head(20).to_string(index=False))
print(f"\nSaved: {out}")
