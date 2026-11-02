# Enterprise MLOps Platform — Real-Data Model Training & Evaluation Report

**Project**: Enterprise MLOps Platform - Customer Churn Prediction  
**Random Seed**: `42`  
**Selected Champion Model**: `XGBoost_Tuned`  

---

## 0. Dataset Provenance & Cryptographic Verification

- **Provenance Status**: **VERIFIED CANONICAL ORIGINAL DATASET**
- **Data Origin**: `canonical_ibm_telco_churn`
- **Raw File Location**: `C:\Users\raksh\.gemini\antigravity\scratch\Enterprise-MLOps-Platform\data\raw\telco_customer_churn.csv`
- **File SHA-256 Checksum**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Verified Canonical Hash Match**: `True`
- **Dataset Shape**: `7043` records, `21` features
- **Target Class Distribution**: `{'No': 5174, 'Yes': 1869}`

> [!NOTE]
> This evaluation was executed strictly against the genuine IBM Telco Customer Churn dataset.
> Automatic synthetic fallback was disabled. Previous synthetic development results are preserved
> separately under `reports/evaluations/synthetic/`.

---

## 1. Candidate Model Benchmark Comparison (Holdout Test Set)

| Model Name | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Brier Score | Latency p95 (ms) | Total Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic_Regression** | **0.8492** | **0.6418** | 0.6289 | 0.5396 | 0.7536 | 0.1668 | 10.69 ms | $43,500.00 |
| **Random_Forest** | **0.8501** | **0.6567** | 0.6372 | 0.5427 | 0.7714 | 0.1610 | 68.28 ms | $41,100.00 |
| **XGBoost_Baseline** | **0.8508** | **0.6655** | 0.6313 | 0.5276 | 0.7857 | 0.1597 | 18.90 ms | $39,850.00 |
| **XGBoost_Tuned** *(Champion)* | **0.8536** | **0.6697** | 0.6486 | 0.5596 | 0.7714 | 0.1611 | 14.51 ms | $40,500.00 |

---

## 2. Champion Model Deep-Dive: `XGBoost_Tuned`

- **Optimal Decision Cutoff**: `0.571` (Yields Calibrated F1: `0.6506`)
- **Confusion Matrix**: TN=`607`, FP=`170`, FN=`64`, TP=`216`
- **Business Cost Breakdown**: $40,500.00 total across test partition ($38.32 / customer)
- **Actual Inference Latency**: Mean=`12.48 ms`, p50=`12.14 ms`, p95=`14.51 ms`

---

## 3. SHAP Explainability & Real-World Churn Risk Factors

Top features driving customer churn risk based on mean absolute SHAP values on real data:

| Rank | Feature | Mean |SHAP Value| Real-World Business Insight |
| :---: | :--- | :---: | :--- |
| 1 | `Contract_Two year` | 0.5470 | High influence on predicted churn probability |
| 2 | `tenure` | 0.4605 | High influence on predicted churn probability |
| 3 | `InternetService_Fiber optic` | 0.3019 | High influence on predicted churn probability |
| 4 | `Contract_One year` | 0.2473 | High influence on predicted churn probability |
| 5 | `MonthlyCharges` | 0.2041 | High influence on predicted churn probability |
| 6 | `PaymentMethod_Electronic check` | 0.1829 | High influence on predicted churn probability |
| 7 | `TotalCharges` | 0.1338 | High influence on predicted churn probability |
| 8 | `InternetService_No` | 0.1237 | High influence on predicted churn probability |

---

## 4. Generated Artifacts & Provenance Lineage

- Serialized Real Champion Pipeline: `models/saved/best_model_pipeline.joblib`
- Reference Drift Baseline (Real Data): `data/processed/reference_dataset.parquet`
- Diagnostic Curves: `reports/figures/roc_curve_comparison.png`, `pr_curve_comparison.png`
- Confusion Matrix Heatmap: `reports/figures/confusion_matrix.png`
- SHAP Visualizations: `reports/figures/shap_summary.png`, `reports/figures/shap_importance_bar.png`
- Metrics JSON Record: `reports/evaluations/training_metrics.json`
- Preserved Synthetic Benchmark: `reports/evaluations/synthetic/synthetic_training_report.md`
