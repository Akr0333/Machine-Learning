# 🎤 Interview Guide — Customer Intelligence & Churn Analytics

## 30-second explanation
> I built an end-to-end customer churn analytics system combining SQL, Python and machine learning. I analyse customer behaviour, engineer features, train a classification model, evaluate it using business-relevant metrics, and expose the analysis through a Streamlit dashboard.

## Why churn prediction?
Churn is a classification problem. The business objective is not simply to maximise accuracy; it is to identify useful high-risk customers while considering the cost of retention outreach and missed churn.

## Why precision and recall?
- **Precision:** among customers predicted as high risk, how many actually churn?
- **Recall:** among customers who actually churn, how many did the model identify?
- The appropriate balance depends on the business cost of false positives versus false negatives.

## What do the current results show?
- Test-set recall: **70.69%**
- Test-set precision: **63.57%**
- Test-set F1: **66.94%**
- Test-set ROC-AUC: **0.739**

These figures come from the reproducible synthetic-data pipeline documented in `RESULTS.md`.

## What would you improve?
- Cross-validation and hyperparameter tuning
- Probability calibration
- SHAP-based explanations
- Cohort and retention analysis
- Model/data drift monitoring
- Production deployment and monitoring

## Data Analyst connection
**SQL → KPI analysis → EDA → segmentation → modelling → visualisation → business recommendation**

This project demonstrates how analytical work can move from a business question to a measurable model and an application.