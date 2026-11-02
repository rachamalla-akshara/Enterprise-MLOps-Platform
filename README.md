# Enterprise MLOps Platform – End-to-End Machine Learning Lifecycle Management

[![MLOps Platform](https://img.shields.io/badge/MLOps-Enterprise_Platform-blue.svg)](https://github.com/rachamalla-akshara/Enterprise-MLOps-Platform)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5.svg?logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitored-E6522C.svg?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-Provisioned-F46800.svg?logo=grafana&logoColor=white)](https://grafana.com)
[![Evidently AI](https://img.shields.io/badge/Evidently_AI-Drift_Detection-6C5CE7.svg)](https://evidentlyai.com)

---

## 1. Project Overview

The **Enterprise MLOps Platform** is an industry-grade, production-hardened Machine Learning lifecycle management platform. It solves the critical operational challenge of deploying, managing, and maintaining machine learning models reliably at scale.

Rather than treating machine learning as static model training, this platform implements a fully automated, closed-loop MLOps architecture:
- **Reproducible Experimentation**: Automated data validation, pipeline transformations, and hyperparameter tuning tracked in MLflow.
- **Model Governance**: Formal Champion/Challenger model registry lifecycle with automated gating gates.
- **Production Serving**: High-performance, asynchronous FastAPI inference engine instrumented with Prometheus telemetry.
- **Inference Persistence**: Real-time logging of prediction payloads, probabilities, and ground-truth feedback into PostgreSQL.
- **Continuous Monitoring & Drift Detection**: Automated statistical data drift and concept drift detection using Evidently AI (Kolmogorov-Smirnov & Chi-Square tests).
- **Automated Retraining Loop**: Event-driven and scheduled retraining triggered upon verified data distribution divergence.
- **Enterprise Deployment**: Multi-stage Docker containers, Docker Compose orchestration for local development, and Kubernetes manifests with Horizontal Pod Autoscaling (HPA) and zero-downtime rolling updates.

---

## 2. Initial ML Use Case: Customer Churn Prediction

The reference implementation targets **Customer Churn Prediction** using an enterprise telecommunications subscription dataset:
- **Task**: Supervised binary classification ($\hat{y} \in \{0, 1\}$).
- **Business Goal**: Identify high-risk churn customers in advance, enabling proactive retention campaigns and minimizing customer lifetime revenue loss.
- **Target Metric Priorities**: PR-AUC and Recall (prioritizing detection of churners given class imbalance where churn rate is $\sim 26\%$), coupled with expected financial cost minimization.
- **Champion Estimator**: Tuned gradient-boosted decision trees (`XGBoost`) encapsulated inside a Scikit-learn `Pipeline` and `ColumnTransformer` guaranteeing zero data leakage.

---

## 3. Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Language & Core** | Python 3.11 | Core runtime for pipelines, APIs, and background daemons |
| **Machine Learning** | Scikit-learn, XGBoost, Optuna | Feature pipelines, baselines, champion model, and HPO |
| **Experiment Tracking** | MLflow | Run tracking, parameter/metric logging, and model registry |
| **Model Serving** | FastAPI, Uvicorn, Pydantic | High-throughput asynchronous REST API for single and batch inference |
| **Database** | PostgreSQL 15 | Structured telemetry: prediction logs, feedback, and drift audits |
| **Object Storage** | MinIO / AWS S3 | Artifact store for MLflow serialized models and Evidently HTML reports |
| **Model Monitoring** | Evidently AI | Non-parametric statistical data and target drift detection |
| **Metrics & Alerts** | Prometheus | System and ML runtime time-series scraping and alerting |
| **Dashboards** | Grafana | Production dashboards for system performance and ML drift telemetry |
| **Containerization** | Docker, Docker Compose | Immutable multi-stage container builds and local stack orchestration |
| **Orchestration** | Kubernetes (k8s) | Declarative deployments, services, ingress, HPA, and health probes |
| **Frontend UI** | React, TypeScript | Operations portal for model evaluation, inference playground, and drift review |
| **CI/CD** | GitHub Actions | Automated linting, testing, security audits, container build, and deployment |

---

## 4. Architectural Overview

```mermaid
flowchart LR
    subgraph Ingestion_Training["Training & Governance"]
        Data[("Raw Data")] --> Validate["Data Validator"]
        Validate --> Preprocess["Feature Pipeline"]
        Preprocess --> XGBoost["XGBoost Training"]
        XGBoost --> MLflow["MLflow Registry\n(Champion/Challenger)"]
    end

    subgraph Serving_Observability["Serving & Telemetry"]
        MLflow -->|"Deploy Champion"| FastAPI["FastAPI Serving Pod"]
        Client["Client / React UI"] -->|"Inference Request"| FastAPI
        FastAPI -->|"Log Predictions"| Postgres[("PostgreSQL")]
        FastAPI -->|"Expose /metrics"| Prometheus["Prometheus"]
        Prometheus --> Grafana["Grafana Dashboards"]
    end

    subgraph Drift_Retraining["Monitoring & Continuous Retraining"]
        Postgres -->|"Sliding Window"| Evidently["Evidently Drift Engine"]
        Evidently -->|"Drift Share > 30%"| RetrainCtrl["Retraining Controller"]
        RetrainCtrl -->|"Trigger Retrain"| Validate
    end
```

---

## 5. Repository Documentation Guide

Complete architectural blueprints and operational specifications are available in the [`docs/`](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs) directory:

- [**System Architecture & Database Schema (`docs/architecture.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/architecture.md): Complete component architecture, repository structure, security design, and PostgreSQL DDL schema.
- [**ML Pipeline & Model Governance (`docs/ml_pipeline.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/ml_pipeline.md): Data ingestion, feature transformations, candidate models, evaluation metrics, and acceptance gates.
- [**MLflow Tracking & Registry Design (`docs/mlflow_design.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/mlflow_design.md): Experiment organization, run metadata schema, model signatures, and promotion policies.
- [**Deployment Architecture (`docs/deployment_design.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/deployment_design.md): Multi-stage Dockerfiles, Docker Compose topology, Kubernetes manifests, HPA, and rolling update strategies.
- [**Monitoring & Drift Detection (`docs/monitoring_design.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/monitoring_design.md): Prometheus metrics catalog, Grafana dashboards, Evidently AI statistical tests, and automated retraining triggers.
- [**Implementation Roadmap & Testing (`docs/implementation_roadmap.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/implementation_roadmap.md): Multi-phase development roadmap, unit/integration/stress testing strategy, and GitHub Actions CI/CD workflows.

---

## 6. Phased Implementation Status

```text
[x] Phase 0: System Architecture, Database Schemas, Design Specs & Environment Setup
[x] Phase 1: Core ML Pipeline, Preprocessing, Baseline Models & Zero-Leakage Architecture
[x] Phase 1.1: Real-Data Canonical Validation & Provenance Verification (COMPLETED)
[ ] Phase 2: MLflow Tracking Server Infrastructure & Dynamic Model Registry Integration (NEXT)
[ ] Phase 3: Production-Grade FastAPI Model Serving Engine & Prediction Telemetry
[ ] Phase 4: Lightweight Docker Multi-Stage Containerization & Compose Orchestration
[ ] Phase 5: Configurable Evidently AI Drift Detection Worker & Automated Retraining Loop
[ ] Phase 6: Prometheus & Grafana Observability Dashboards & Alert Rules
[ ] Phase 7: Kubernetes Orchestration (Manifests, HPA, Ingress & Health Probes)
[ ] Phase 8: GitHub Actions CI/CD Automation & Security Scanning
[ ] Phase 9: React Operations Dashboard & Inference Playground
```

---

## 7. Real-Data Empirical Benchmark Results (Holdout Test Set)

The platform was validated and benchmarked strictly against the genuine, canonical **IBM Telco Customer Churn** dataset:
- **Dataset Provenance Status**: Verified Canonical Dataset (`canonical_ibm_telco_churn`)
- **Raw File Checksum (SHA-256)**: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Dataset Dimensions**: 7,043 rows, 21 columns
- **Ground Truth Target Distribution**: Retained (`No`): 5,174 (73.46%), Churned (`Yes`): 1,869 (26.54%)
- **Untouched Holdout Test Partition**: 15% stratified split ($N = 1,057$ records)
- **Synthetic Development Baseline**: Preserved separately under [`reports/evaluations/synthetic/`](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/reports/evaluations/synthetic/synthetic_training_report.md)

### Model Benchmark Comparison (Holdout Test Set):

| Model Candidate | ROC-AUC | PR-AUC | F1-Score | Recall | Precision | Brier Score | Latency p95 | Business Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.8492 | 0.6418 | 0.6289 | 0.7536 | 0.5396 | 0.1668 | 10.69 ms | $43,500.00 |
| **Random Forest Classifier** | 0.8501 | 0.6567 | 0.6372 | 0.7714 | 0.5427 | 0.1610 | 68.28 ms | $41,100.00 |
| **XGBoost Classifier (Baseline)** | 0.8508 | 0.6655 | 0.6313 | **0.7857** | 0.5276 | **0.1597** | 18.90 ms | **$39,850.00** |
| **XGBoost Classifier (Tuned Champion)** | **0.8536** | **0.6697** | **0.6486** | 0.7714 | **0.5596** | 0.1611 | **14.51 ms** | $40,500.00 |

*Champion Model*: `XGBoost_Tuned` (`n_estimators=200`, `max_depth=3`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`).  
*Optimal Calibrated Cutoff*: `0.571` (yields calibrated F1: `0.6506`).  
*Key Real-World Churn Drivers (SHAP)*: `Contract_Two year` (0.5470), `tenure` (0.4605), `InternetService_Fiber optic` (0.3019), `Contract_One year` (0.2473), and `MonthlyCharges` (0.2041).

---

## 8. Development & Pipeline Quickstart

### 1. Environment Setup
```bash
cp .env.example .env
```

### 2. Run Test Suite (Unit Tests, Provenance & Data Leakage Tests)
```bash
pytest tests/ -v
# Output: 20 passed, 0 failed in ~4.5s
```

### 3. Execute End-to-End Real-Data Training Pipeline
```bash
python -m src.models.train
```
This executes:
- Strict dataset provenance and SHA-256 validation (zero silent fallback).
- Pydantic schema validation and defensive missing value handling.
- Exploratory data analysis with visualization generation.
- Stratified partitioning (70% train, 15% val, 15% untouched test).
- Preprocessing fit strictly on $X_{train}$ (zero data leakage).
- Model training (Logistic Regression, Random Forest, XGBoost baseline, XGBoost tuned).
- Empirical metric evaluation and actual latency benchmarking.
- Real-data SHAP feature attribution generation.
- Persistence of champion pipeline to `models/saved/best_model_pipeline.joblib`.
- Comprehensive markdown report to `reports/evaluations/training_report.md`.
