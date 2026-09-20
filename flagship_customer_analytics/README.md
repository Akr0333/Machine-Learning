# 📈 Customer Intelligence & Churn Analytics

An end-to-end **Data Analytics + Machine Learning** project designed around a realistic business question:

> Which customers are likely to churn, why are they leaving, and where should the business focus retention efforts?

## 🎯 Business Goals

- Monitor customer revenue and engagement
- Identify churn patterns by segment
- Quantify important churn drivers
- Predict customers at risk of leaving
- Turn model output into actionable retention insights

## 🔄 End-to-End Workflow

`Raw Data → SQL Analysis → Python EDA → Feature Engineering → ML → Evaluation → Business Insights → Dashboard`

## 🧰 Stack

**Analytics:** SQL, Pandas, NumPy  
**Visualisation:** Matplotlib, Seaborn, Plotly  
**Machine Learning:** Scikit-learn  
**Dashboard:** Streamlit  
**Engineering:** Git, GitHub, Pytest

## 📊 Analytics Layer

The SQL analysis covers:

- Revenue by customer segment
- Monthly recurring revenue
- Churn rate
- Average revenue per customer
- Tenure analysis
- Contract and payment-method analysis
- High-value customers at risk

## 🤖 ML Layer

Candidate models:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

For churn-risk use cases, the project explicitly considers **recall and ROC-AUC alongside accuracy**, because missing a genuinely at-risk customer can be costly.

## 💼 Business Output

The final output is designed to answer:

1. Who is churning?
2. Which customer segments have the highest risk?
3. Which factors are associated with churn?
4. How much revenue is exposed?
5. Which customers should be prioritised for retention?

## 🚀 Roadmap

- [x] Project architecture
- [x] SQL analytics layer
- [x] Python analysis layer
- [x] ML training workflow
- [x] Dashboard structure
- [ ] Add production customer dataset
- [ ] Add model explainability
- [ ] Deploy dashboard
- [ ] Add automated model monitoring

This project demonstrates the complete path from **business question → data → analysis → prediction → decision support**.
