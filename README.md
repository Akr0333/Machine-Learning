# 🤖 Machine Learning Portfolio

A practical Machine Learning portfolio covering **data preparation, analytics, prediction, evaluation and deployment**.

## 📈 Flagship: Customer Intelligence & Churn Analytics

An end-to-end **Data Analytics + ML** system answering a realistic business question:

> Which customers are likely to churn, why are they leaving, and where should retention efforts focus?

### Workflow

**Business Question → SQL → Python EDA → Feature Engineering → ML → Evaluation → Dashboard → Business Insight**

### Includes

- SQL business analytics and KPIs
- Reproducible customer-data generation
- Logistic Regression churn model
- Precision, recall, F1 and ROC-AUC evaluation
- Revenue-at-risk analysis
- Streamlit dashboard
- High-value customer risk table
- Reproducible project documentation

### Run it

```bash
cd flagship_customer_analytics
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
streamlit run app.py
```

See [flagship_customer_analytics/README.md](flagship_customer_analytics/README.md) for the full business and technical breakdown.

---

## 🏠 House Price Prediction

An end-to-end regression system with preprocessing, model training, evaluation, model persistence and Streamlit inference.

### ML workflow

**Problem → Data → EDA → Preprocessing → Features → Model → Evaluation → Inference**

## 🧰 Stack

**Python:** Pandas • NumPy
**ML:** Scikit-learn • Joblib
**Analytics:** SQL • Pandas
**Visualisation:** Matplotlib • Seaborn • Plotly
**Apps:** Streamlit
**Testing:** Pytest

## 🔮 Roadmap

- Add SHAP-based explainability
- Add Docker and CI/CD
- Add automated model monitoring
- Deploy flagship dashboard
- Add real-world datasets

⭐ **Analyse → Model → Explain → Deploy**