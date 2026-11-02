# ML Pipeline Architecture & Model Governance Specification

## 1. Problem Formulation: Customer Churn Prediction

Customer Churn Prediction is formulated as a supervised binary classification task:
$$\hat{y} \in \{0, 1\}$$
where $y = 1$ denotes a customer terminating their subscription within the observation window, and $y = 0$ denotes customer retention.

### Business Objective & Cost Matrix
In enterprise subscription environments, churned customers represent lost recurring revenue (Customer Lifetime Value, $LTV$). Interventions (promotions, outreach) incur an intervention cost ($C_{int}$).

$$\text{Total Business Cost} = FN \cdot LTV + FP \cdot C_{int}$$

Because $LTV \gg C_{int}$, prioritizing Recall and Precision-Recall Area Under the Curve (PR-AUC) over raw accuracy is mandatory. Raw accuracy is misleading due to class imbalance (~26% churn rate).

---

## 2. Dataset Schema & Feature Hierarchy

The baseline dataset utilizes the industry-standard Telco Customer Churn dataset with 20 input attributes:

| Feature Category | Features | Data Type | Preprocessing Strategy |
| :--- | :--- | :--- | :--- |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical / Binary | Binary encoding & One-Hot Encoding |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Multi-category nominal | One-Hot Encoding (drop first to prevent multicollinearity) |
| **Account Details**| `Contract`, `PaperlessBilling`, `PaymentMethod` | Nominal / Categorical | One-Hot Encoding |
| **Financial / Usage**| `tenure`, `MonthlyCharges`, `TotalCharges` | Continuous Numerical | Median imputation for missing values, Robust/Standard scaling |
| **Target** | `Churn` (`Yes`/`No`) | Binary | Transformed to integer `{0, 1}` |

---

## 3. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Ingestion & Contract Enforcement"]
        RawCSV["Raw Input Data\n(Local CSV or S3 Storage)"]
        PydanticSchema["Pydantic Contract Validator\n(Type checking, Range validation)"]
        Splitter["Stratified Splitter\n(Train: 70%, Val: 15%, Test: 15%)"]
    end

    subgraph Preprocessing["2. Preprocessing & Feature Pipeline"]
        NumericPipe["Numerical Pipeline\n(Median Imputer + StandardScaler)"]
        CategoricalPipe["Categorical Pipeline\n(Missing Imputer + OneHotEncoder)"]
        ColumnTransformer["Scikit-Learn ColumnTransformer\n(Encapsulated, zero data leakage)"]
    end

    subgraph Modeling["3. Multi-Model Training & Tuning"]
        BaselineModels["Baseline Models\n(Logistic Regression, Random Forest)"]
        XGBoostModel["Production Candidate\n(XGBoost with scale_pos_weight)"]
        HPO["Hyperparameter Optimization\n(Optuna / Bayesian Search)"]
    end

    subgraph Evaluation["4. Comprehensive Evaluation & Explainability"]
        MetricsEngine["Metrics Calculator\n(ROC-AUC, PR-AUC, F1, Brier Score)"]
        CostAnalysis["Financial Cost Matrix Evaluator"]
        SHAPEngine["SHAP TreeExplainer\n(Feature importances, Beeswarm summary)"]
    end

    subgraph Governance["5. Automated Gating & MLflow Registration"]
        MLflowLogger["MLflow Run Logger\n(Params, Metrics, Artifacts, Signatures)"]
        GatingRule["Gating Engine\n(Holdout Test Thresholds & Challenger Check)"]
        RegistryAction["MLflow Model Registry\n(Tag as Candidate or Champion)"]
    end

    RawCSV --> PydanticSchema
    PydanticSchema --> Splitter
    Splitter --> NumericPipe
    Splitter --> CategoricalPipe
    NumericPipe --> ColumnTransformer
    CategoricalPipe --> ColumnTransformer

    ColumnTransformer --> BaselineModels
    ColumnTransformer --> XGBoostModel
    XGBoostModel --> HPO
    HPO --> MetricsEngine

    MetricsEngine --> CostAnalysis
    MetricsEngine --> SHAPEngine
    MetricsEngine --> MLflowLogger
    CostAnalysis --> MLflowLogger
    SHAPEngine --> MLflowLogger

    MLflowLogger --> GatingRule
    GatingRule -->|"Meets Acceptance Criteria"| RegistryAction
    GatingRule -->|"Fails Acceptance Criteria"| Reject["Audit Logged & Quarantined"]
```

---

## 4. Zero Data-Leakage Preprocessing Design

To prevent target leakage and data leakage between splits:
1. **Splitting Precedes Transformation**: Train, validation, and test splits are generated *before* computing any statistics (means, standard deviations, categorical frequencies).
2. **Encapsulated Pipelines**: All preprocessing steps are packaged inside a `sklearn.pipeline.Pipeline` or `imblearn.pipeline.Pipeline`.
3. **Artifact Bundling**: The fitted preprocessor and the trained estimator are serialized together as a single unified inference pipeline object (`pipeline.joblib`). The API service receives raw JSON, passes it directly into `pipeline.predict_proba()`, guaranteeing identical feature engineering in serving as in training.

---

## 5. Candidate Model Suite & Optimization

1. **Baseline Model**:
   - `LogisticRegression(class_weight='balanced', max_iter=1000)`: Establishes linear separability benchmark and feature weight interpretability.
   - `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')`: Establishes non-linear tree ensemble baseline.

2. **Primary Champion Model**:
   - `XGBClassifier`: Gradient boosted decision trees optimized with:
     - `scale_pos_weight = count(negative) / count(positive)` to handle churn class imbalance.
     - `eval_metric = ['auc', 'aucpr', 'logloss']`.
     - Early stopping on the validation set to eliminate overfitting.
     - Optuna hyperparameter optimization over `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and `reg_lambda`.

---

## 6. Automated Model Acceptance Criteria (Gating Protocol)

Before any trained model is eligible for registration or staging promotion, it must clear the **Automated Gating Engine**:

| Gate Identifier | Metric / Check | Minimum Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| **GATE-01** | Test ROC-AUC | $\ge 0.830$ | Ensures strong discriminative separation across decision thresholds |
| **GATE-02** | Test PR-AUC | $\ge 0.650$ | Validates precision-recall balance on minority churn class |
| **GATE-03** | Test F1-Score (Optimal Cutoff) | $\ge 0.600$ | Guarantees balanced operational performance |
| **GATE-04** | Inference Latency Benchmark | $\le 25\text{ ms}$ (p95, batch=1) | Guarantees service level objectives under real-time load |
| **GATE-05** | Challenger Improvement | $\Delta \text{PR-AUC} > +0.01$ | Prevents unnecessary model churn unless statistically superior |
| **GATE-06** | Model Schema Signature | Input/Output contract verified | Ensures FastAPI serving compatibility without runtime crashes |

Models failing any gate are logged with error status in MLflow, tagged as `QUARANTINED`, and will never be pushed to the active serving stage.
