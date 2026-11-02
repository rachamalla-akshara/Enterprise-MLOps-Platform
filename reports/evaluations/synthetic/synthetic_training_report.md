# Enterprise MLOps Platform — Model Training & Evaluation Report

**Project**: Enterprise MLOps Platform - Customer Churn Prediction  
**Random Seed**: `42`  
**Selected Champion Model**: `XGBoost_Tuned`  

---

## 1. Candidate Model Benchmark Comparison (Holdout Test Set)

| Model Name | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Brier Score | Latency p95 (ms) | Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic_Regression** | **0.8559** | **0.7110** | 0.7087 | 0.6329 | 0.8052 | 0.1582 | 10.20 ms | $42,150.00 |
| **Random_Forest** | **0.8550** | **0.7039** | 0.6974 | 0.6624 | 0.7364 | 0.1574 | 79.31 ms | $52,550.00 |
| **XGBoost_Baseline** | **0.8532** | **0.7090** | 0.7040 | 0.6123 | 0.8281 | 0.1691 | 20.36 ms | $39,150.00 |
| **XGBoost_Tuned** *(Champion)* | **0.8566** | **0.7155** | 0.7032 | 0.6225 | 0.8080 | 0.1697 | 13.82 ms | $42,050.00 |

---

## 2. Champion Model Deep-Dive: `XGBoost_Tuned`

- **Optimal Decision Cutoff**: `0.558` (Yields Calibrated F1: `0.7082`)
- **Confusion Matrix**: TN=`537`, FP=`171`, FN=`67`, TP=`282`
- **Business Cost Breakdown**: $42,050.00 total across test partition ($39.78 / customer)
- **Actual Inference Latency**: Mean=`12.36 ms`, p50=`12.08 ms`, p95=`13.82 ms`

---

## 3. SHAP Explainability & Top Churn Risk Factors

Top features driving customer churn risk based on mean absolute SHAP values:

| Rank | Feature | Mean |SHAP Value| Impact on Churn Risk |
| :---: | :--- | :---: | :--- |
| 1 | `tenure` | 0.8487 | High influence on predicted probability |
| 2 | `Contract_Two year` | 0.5823 | High influence on predicted probability |
| 3 | `Contract_One year` | 0.3758 | High influence on predicted probability |
| 4 | `MonthlyCharges` | 0.3457 | High influence on predicted probability |
| 5 | `InternetService_Fiber optic` | 0.2625 | High influence on predicted probability |
| 6 | `PaymentMethod_Electronic check` | 0.1315 | High influence on predicted probability |
| 7 | `TotalCharges` | 0.1011 | High influence on predicted probability |
| 8 | `OnlineSecurity_Yes` | 0.0659 | High influence on predicted probability |

---

## 4. Model Artifacts & Diagnostics Generated

- Serialized Champion Pipeline: `models/saved/best_model_pipeline.joblib`
- Reference Drift Baseline: `data/processed/reference_dataset.parquet`
- Diagnostic Curves: `reports/figures/roc_curve_comparison.png`, `pr_curve_comparison.png`
- Confusion Matrix Heatmap: `reports/figures/confusion_matrix.png`
- SHAP Visualizations: `reports/figures/shap_summary.png`, `reports/figures/shap_importance_bar.png`
- Metrics JSON Database: `reports/evaluations/training_metrics.json`
