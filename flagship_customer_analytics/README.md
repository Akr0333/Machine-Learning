# 📈 Customer Intelligence & Churn Analytics

**End-to-end Data Analytics + Machine Learning portfolio project**

> **Business goal:** identify customers at risk of churn, understand the drivers behind that risk, and turn model output into actionable retention insights.

## 🎯 Why this project matters

A model score alone is not enough for a business analyst or data scientist.

This project connects:

**SQL → Data Analysis → Machine Learning → Evaluation → Business Insights → Dashboard**

## 🔄 End-to-End Workflow

1. Generate / prepare customer data
2. Analyse KPIs with SQL
3. Clean and explore data with Python
4. Engineer predictive features
5. Train a churn classification model
6. Evaluate precision, recall, F1 and ROC-AUC
7. Segment customers by risk
8. Estimate revenue at risk
9. Explore results through a Streamlit dashboard

## 🧠 Skills Demonstrated

### Data Analytics
- Pandas / NumPy
- EDA
- KPI analysis
- Segmentation
- Revenue-at-risk analysis

### SQL
- JOINs
- CTEs
- Aggregations
- CASE expressions
- Window functions

### Machine Learning
- Classification
- Feature engineering
- Logistic Regression
- Model evaluation
- Probability-based risk scoring

### Engineering
- Streamlit
- Pytest
- Docker
- GitHub Actions
- Reproducible project structure

## 📊 Key Business Outputs

The dashboard is designed around questions such as:

- How many customers are high-risk?
- Which customer segments contribute most to potential churn?
- Which characteristics are associated with higher risk?
- What revenue is potentially exposed?
- Where should a retention team prioritise investigation?

## 📁 Structure

```text
flagship_customer_analytics/
├── app.py
├── train.py
├── explain.py
├── generate_data.py
├── sql/
├── data/
├── models/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

## ▶️ Quick Start

```bash
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

## 💼 Interview Talking Points

Be ready to explain:

- Why churn is a classification problem
- Why accuracy alone can be misleading
- Precision vs recall for retention use cases
- How feature engineering affects model quality
- How model probabilities become business risk segments
- How SQL and Python work together in an analyst workflow

## 🔮 Next Improvements

- SHAP model explanations
- Hyperparameter optimisation
- Model monitoring
- Drift detection
- Interactive cohort analysis
- Production deployment

**Portfolio principle: Business Question → Evidence → Model → Insight → Action**
