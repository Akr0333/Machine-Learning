# Customer Churn Prediction 📉

An end-to-end machine learning project that predicts whether a customer is likely to churn.

## 🎯 Objective
Build a classification pipeline that converts customer information into a churn prediction and probability score.

## 🧠 ML Workflow
1. Data loading
2. Data quality checks
3. Exploratory data analysis
4. Feature preprocessing
5. Train/test split
6. Baseline Logistic Regression
7. Random Forest comparison
8. Evaluation using precision, recall, F1 and ROC-AUC
9. Feature importance
10. Model export

## 🛠️ Stack
Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-learn • Joblib

## 📁 Structure
```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── churn_analysis.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── models/
│   └── .gitkeep
└── reports/
    └── README.md
```

## 📊 Key Questions
- Which customer groups churn most?
- Which features are strongest churn indicators?
- How does the model perform on unseen data?
- Which metric matters most when churn has a business cost?

## 🚀 Future Improvements
Hyperparameter tuning, cross-validation, probability calibration, SHAP explanations and deployment with Streamlit.
