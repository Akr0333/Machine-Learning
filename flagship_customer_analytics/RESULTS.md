# 📊 Verified Results — Customer Intelligence & Churn Analytics

These results are reproducible from the project's synthetic dataset generator (`src/generate_data.py`) using the training pipeline in `src/train.py`.

## Dataset
- Customers: **1,200**
- Churn rate: **48.25%**
- Average monthly revenue: **$77.09**
- Revenue associated with churned customers: **$45,182.02**

## Test-set model performance
The project uses an 80/20 stratified train/test split with `random_state=42` and a class-weighted Logistic Regression pipeline.

| Metric | Result |
|---|---:|
| Accuracy | **66.25%** |
| Precision | **63.57%** |
| Recall | **70.69%** |
| F1-score | **66.94%** |
| ROC-AUC | **0.739** |

### Interpretation
The model identifies approximately **70.69% of churn cases in the held-out test set** (recall). Precision is **63.57%**, meaning that roughly 64% of customers predicted as churners were churners in this test set.

ROC-AUC of **0.739** indicates useful ranking ability on this synthetic demonstration dataset.

> ⚠️ This is a portfolio demonstration using synthetic data. These numbers should not be presented as production performance or as evidence about a real company's customers.

## Reproduce
```bash
python src/generate_data.py
python src/train.py
```

The generator is seeded with `42`, and the model evaluation uses a stratified 80/20 split with `random_state=42`, so the reported figures are reproducible with the current code.