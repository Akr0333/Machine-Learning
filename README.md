# Machine Learning 🤖

A practical Machine Learning portfolio covering the full workflow from **data preparation to model evaluation**.

## 🚀 Featured Project: House Price Prediction

An end-to-end regression system that predicts house prices from property characteristics. It includes reproducible data generation, preprocessing, model training, evaluation, model persistence, inference and a Streamlit web app.

### Project structure

```text
Machine-Learning/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── train.py
│   └── predict.py
└── tests/
    └── test_pipeline.py
```

### Quick start

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python -m src.generate_data
python -m src.train
streamlit run app.py
pytest -q
```

### ML workflow

**Problem → Data → EDA → Preprocessing → Features → Model → Evaluation → Inference**

The project uses a scikit-learn `Pipeline` and `ColumnTransformer`, so preprocessing and the model are saved together and inference uses exactly the same transformations as training.

### Features

- Area in square feet
- Bedrooms and bathrooms
- Property age
- Distance from city centre
- Location
- Parking availability
- Furnishing status

### Metrics

The training script reports **MAE, RMSE and R²** on a held-out test set.

### Stack

**Python:** Pandas • NumPy  
**ML:** Scikit-learn • Joblib  
**App:** Streamlit  
**Testing:** Pytest

### Future improvements

- Replace synthetic data with a real housing dataset
- Add cross-validation and hyperparameter tuning
- Add SHAP-based explainability
- Add Docker and CI/CD
- Deploy the app to the cloud

⭐ **Experiment → Evaluate → Improve → Build.**
