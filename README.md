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
- [**MLflow Implementation & Operations (`docs/mlflow_implementation.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/mlflow_implementation.md): Complete implementation guide covering SQLite tracking, cloudpickle logging, promotion gating, rollback, and dynamic alias resolution.
- [**FastAPI Model Serving & Telemetry (`docs/fastapi_serving.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/fastapi_serving.md): High-throughput REST API serving engine, dynamic MLflow alias loading, atomic model reloads, and telemetry logging.
- [**Docker Deployment & Orchestration (`docs/docker_deployment.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/docker_deployment.md): Multi-stage container builds, Docker Compose topology, persistent volume architecture, and MLflow migration tools.
- [**Deployment Architecture (`docs/deployment_design.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/deployment_design.md): Multi-stage Dockerfiles, Docker Compose topology, Kubernetes manifests, HPA, and rolling update strategies.
- [**Monitoring & Drift Detection (`docs/monitoring_design.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/monitoring_design.md): Prometheus metrics catalog, Grafana dashboards, Evidently AI statistical tests, and automated retraining triggers.
- [**Evidently AI Drift Monitoring & Retraining Guide (`docs/drift_monitoring.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/drift_monitoring.md): Complete guide to Phase 5 drift detection, privacy-aware telemetry buffer, statistical tests, delayed ground-truth evaluation, and retraining policies.
- [**Prometheus Observability & Grafana Dashboards (`docs/observability.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/observability.md): Complete specification of Prometheus metrics catalog, 12 alert rules across 3 tiers, 4 auto-provisioned Grafana dashboards, and Docker topology.
- [**Implementation Roadmap & Testing (`docs/implementation_roadmap.md`)**](file:///C:/Users/raksh/.gemini/antigravity/scratch/Enterprise-MLOps-Platform/docs/implementation_roadmap.md): Multi-phase development roadmap, unit/integration/stress testing strategy, and GitHub Actions CI/CD workflows.

---

## 6. Phased Implementation Status

```text
[x] Phase 0: System Architecture, Database Schemas, Design Specs & Environment Setup
[x] Phase 1: Core ML Pipeline, Preprocessing, Baseline Models & Zero-Leakage Architecture
[x] Phase 1.1: Real-Data Canonical Validation & Provenance Verification (COMPLETED)
[x] Phase 2: MLflow Tracking Server Infrastructure & Dynamic Model Registry Integration (COMPLETED)
[x] Phase 3: Production-Grade FastAPI Model Serving Engine & Prediction Telemetry (COMPLETED)
[x] Phase 4: Lightweight Docker Multi-Stage Containerization & Compose Orchestration (COMPLETED)
[x] Phase 5: Evidently AI Data Drift Detection, Model Monitoring & Retraining Controller (COMPLETED)
[x] Phase 6: Prometheus & Grafana Observability Dashboards & Alert Rules (COMPLETED)
[ ] Phase 7: Kubernetes Orchestration (Manifests, HPA, Ingress & Health Probes) (NEXT)
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

### 2. Run Test Suite (Unit, Leakage, Validation, MLflow Registry & Serving Tests)
```bash
python -m pytest tests/ -v
# Output: 47 passed, 0 failed in ~1.5m
```

### 3. Execute End-to-End Real-Data Training & MLflow Pipeline
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
- Empirical metric evaluation (separate train, val, and test metrics).
- MLflow run tracking: parameters, separate split metrics, diagnostic plots, and SHAP explainability.
- Pipeline encapsulation with cloudpickle serialization and schema signatures.
- Model registry registration under `customer-churn-classifier`.
- Acceptance gating: candidate evaluation, automatic champion promotion (`@champion`) and runner-up challenger assignment (`@challenger`).
- Dynamic alias verification: loading via `models:/customer-churn-classifier@champion` and validating prediction parity.
- Comprehensive markdown report exported to `reports/evaluations/training_report.md`.

### 4. Launch Local MLflow Tracking UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open `http://localhost:5000` to inspect runs, metric curves, model artifacts, and registry aliases.

### 5. Launch FastAPI Model Serving Engine
```bash
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Example Prediction Request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.2
  }'
```

#### Example Response:
```json
{
  "request_id": "8aa165ea-edb8-4033-ae10-229062ff2067",
  "prediction": "No Churn",
  "churn_probability": 0.4812,
  "decision_threshold": 0.571,
  "model_name": "customer-churn-classifier",
  "model_version": "4",
  "inference_latency_ms": 22.88,
  "prediction_timestamp": "2026-11-02T14:19:03.123456+00:00"
}
```

#### Trigger Protected Model Reload:
```bash
curl -X POST http://127.0.0.1:8000/model/reload \
  -H "Content-Type: application/json" \
  -H "X-Admin-API-Key: dev-admin-secret-key-phase3" \
  -d '{"alias": "champion"}'
```

---

## 9. Docker Containerization & Compose Orchestration

### 1. Prerequisites
- Docker Engine 24.0+ / Docker Desktop
- Docker Compose v2.20+ (included with Docker Desktop)

### 2. Validate Compose Specification
```bash
docker compose config
```

### 3. Build & Launch the Complete MLOps Stack
```bash
# Build multi-stage image and launch core services (PostgreSQL, MLflow, FastAPI)
docker compose up -d --build

# Optional: Launch with MinIO S3 object storage profile
docker compose --profile storage up -d
```

### 4. Service Endpoints in Docker
- **FastAPI Serving Engine**: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)
- **MLflow Tracking UI**: `http://localhost:5000`
- **PostgreSQL Database**: `localhost:5432` (`mlops_user` / `mlops_password` / `mlops_db`)
- **MinIO Console** *(Storage profile only)*: `http://localhost:9001` (`minioadmin` / `minioadmin`)

### 5. Migrate Existing Model Registry into PostgreSQL
To synchronize the local SQLite registry (`mlflow.db`) and artifacts into the containerized PostgreSQL database with prediction parity verification:
```bash
python scripts/migrate_mlflow_registry.py \
  "sqlite:///mlflow.db" \
  "postgresql+psycopg2://mlops_user:mlops_password@localhost:5432/mlops_db"
```

### 6. Phase 4.2 Verified Runtime Architecture
- **Single Uvicorn Worker**: Running with `--workers 1` guarantees that in-memory model state and dynamic reload operations do not suffer from split-brain divergence across process boundaries. Horizontal concurrency is scaled across container replicas behind a load balancer.
- **MLflow 3.x Host Authorization**: Configured with `--allowed-hosts "*"` to permit internal Docker bridge network communication (`http://mlflow:5000`) without DNS rebinding protection rejections.
- **Artifact Volume Storage**: Mounted read-write (`mlflow_artifacts:/app/mlruns`) to permit model metadata caching and dynamic alias resolution.
- **Zero-PII Telemetry**: Prediction telemetry is continuously logged to `public.prediction_logs` in PostgreSQL without sensitive customer features.

### 7. Compose Teardown
```bash
# Stop containers while preserving persistent volumes
docker compose down

# Stop containers and erase all volume data (CAUTION)
docker compose down -v
```

---

## 10. Evidently AI Data Drift Detection & Model Monitoring (Phase 5)

Phase 5 introduces continuous production monitoring, non-parametric statistical drift detection, data quality auditing, delayed ground-truth performance evaluation, and controlled retraining recommendations without human privacy violations.

### 1. Architectural Highlights & Privacy Invariants

- **Zero-PII Ephemeral Telemetry Buffer**: Customer features are never logged permanently. Predictions are stripped of identifiers; raw feature observations are temporarily held in an ephemeral bounded buffer table (`monitoring.feature_observations`, default capacity: 5,000 records, retention TTL: 72 hours) with automatic FIFO eviction.
- **Persistent Aggregate Audits**: Statistical metrics, drift shares, p-values, distance metrics, and data quality summaries are permanently recorded across 7 dedicated tables in the `monitoring` PostgreSQL schema.
- **Champion Protection Invariant**: Statistical drift detection or performance degradation triggers explicit **retraining recommendations** with remediation playbooks. Drift detection **never automatically replaces, aliases, or promotes models** directly to production.

```mermaid
flowchart TD
    subgraph Serving["FastAPI Inference Engine"]
        Req["Inference Request\n(Pydantic validated)"] --> Predict["Champion Model Predict"]
        Predict --> PredLog[("public.prediction_logs\n(request_id, latency, prob)")]
        Predict --> Buffer[("monitoring.feature_observations\n(TTL: 72h / FIFO: 5k)")]
    end

    subgraph MonitoringWorker["Evidently Monitoring Worker (Daemon / CLI / API)"]
        Buffer -->|"Sliding Window (N >= 50)"| Quality["Data Quality Monitor\n(Missingness, ranges, novel classes)"]
        Quality --> Drift["Drift Detector\n(KS test, Wasserstein, Chi-Square, TVD)"]
        RefData[("data/processed/\nreference_dataset.parquet")] --> Drift
        PredLog --> PredDrift["Prediction Monitor\n(Probability KS & Churn Rate Divergence)"]
        
        Feedback[("monitoring.ground_truth_feedback\n(delayed actual churn)")] --> Perf["Performance Monitor\n(ROC-AUC, PR-AUC, F1, Brier)"]
        
        Drift --> RecEngine["Retraining Recommendation Engine\n(Cooldown: 24h)"]
        Perf --> RecEngine
        PredDrift --> RecEngine
        
        RecEngine --> RecLog[("monitoring.retraining_recommendations")]
        Drift --> Reports["Interactive HTML & JSON Reports\n(reports/monitoring/)"]
    end
```

### 2. Statistical Drift Detection Methodology

| Feature Type | Primary Statistical Test | Fallback / Complementary Metric | Drift Threshold |
| :--- | :--- | :--- | :--- |
| **Numerical** (`tenure`, `MonthlyCharges`, `TotalCharges`) | Two-Sample Kolmogorov-Smirnov (KS) test | 1D Wasserstein Distance ($L_1$) | $p < 0.05$ (KS) or $W > 0.1$ |
| **Categorical** (16 contract/service features) | Pearson's Chi-Square Test ($\ge 5$ observations/cell) | Total Variation Distance (TVD) | $p < 0.05$ ($\chi^2$) or $\text{TVD} > 0.1$ |
| **Dataset Drift Gate** | Drifted Feature Share | Overall feature pool ($N=19$) | $\text{Drift Share} \ge 30\%$ (6+ drifted features) |
| **Prediction Drift** | KS Test on Output Probabilities | Divergence from Reference Churn Rate (26.54%) | $p < 0.05$ or $\|\Delta \text{Churn Rate}\| > 15\%$ |

### 3. Monitoring API Endpoints

FastAPI exposes authenticated monitoring endpoints under `/monitoring`:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/monitoring/status` | Current monitoring status, sample counts, active window, and last run time |
| `GET` | `/monitoring/latest` | Latest drift report, data quality summary, and retraining recommendations |
| `POST` | `/monitoring/run` | Trigger an on-demand monitoring cycle with HTML and JSON report export |
| `POST` | `/monitoring/feedback` | Ingest delayed ground-truth labels matched against inference `request_id` |

#### Example: Ingest Ground-Truth Feedback
```bash
curl -X POST http://localhost:8000/monitoring/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedbacks": [
      {
        "request_id": "8aa165ea-edb8-4033-ae10-229062ff2067",
        "actual_churn": 1,
        "notes": "Customer cancelled service after promotional rate expired"
      }
    ]
  }'
```

### 4. Running the Monitoring Worker CLI

The standalone monitoring worker can be invoked locally or inside Docker:

```bash
# Check status and observation counts
python -m src.monitoring.worker --status

# Execute a single monitoring run
python -m src.monitoring.worker --run-once

# Ingest test observation batch from raw CSV
python -m src.monitoring.worker --ingest-observations data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --limit 100

# Ingest ground-truth labels from CSV
python -m src.monitoring.worker --ingest-ground-truth data/raw/ground_truth_batch.csv

# Run as background daemon with 3600-second interval
python -m src.monitoring.worker --daemon --interval 3600

# Purge expired ephemeral observations manually
python -m src.monitoring.worker --purge
```

### 5. Retraining Recommendation Policies

When drift or degradation is confirmed, the engine issues a structured recommendation with specific root causes:

1. **`DATASET_DRIFT` Trigger**: Triggered when drifted feature share $\ge 30\%$. Recommended action: Feature pipeline audit and scheduled retraining on recent data window.
2. **`PERFORMANCE_DROP` Trigger**: Triggered when evaluated holdout ROC-AUC drops below $0.78$ or F1 drops below $0.55$. Recommended action: Immediate model retraining and champion/challenger re-evaluation.
3. **`PREDICTION_DRIFT` Trigger**: Triggered when predicted churn rate shifts by $> 15\%$ from the training prior ($26.54\%$) or output probability distribution drifts significantly ($p < 0.05$).
4. **`DATA_QUALITY_FAIL` Trigger**: Triggered when missing value rates or invalid category rates exceed acceptable thresholds.
5. **Cooldown Enforcement**: Once a retraining recommendation is recorded, an automatic **24-hour cooldown** prevents redundant alert spam.

---

## 11. Phase 6: Prometheus Observability, Grafana Dashboards & Alert Rules

Phase 6 implements a comprehensive, production-hardened observability layer providing real-time visibility into HTTP serving performance, machine learning model inference latencies, champion governance metadata, and statistical data/prediction drift detection.

```mermaid
flowchart LR
    Client["Client Traffic"] -->|"HTTP Requests"| FastAPI["FastAPI Engine (:8000)"]
    FastAPI -->|"Prometheus Middleware"| Metrics["GET /metrics"]
    Prometheus["Prometheus Server (:9090)"] -->|"Scrape 5s"| Metrics
    Prometheus -->|"Evaluate 15s"| Alerts["monitoring/prometheus/alerts.yml\n(12 Rules across 3 Tiers)"]
    Grafana["Grafana Server (:3000)"] -->|"Query PromQL"| Prometheus
    Grafana --> DashA["Dashboard A: API Overview"]
    Grafana --> DashB["Dashboard B: ML Inference"]
    Grafana --> DashC["Dashboard C: Model Monitoring"]
    Grafana --> DashD["Dashboard D: Infrastructure"]
```

### 1. Prometheus Metrics Catalog

All metrics adhere to strict low-cardinality label rules and guarantee zero leakage of customer PII, request identifiers, or sensitive payload features:

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `http_requests_total` | Counter | `method`, `endpoint`, `status_code` | Total HTTP requests processed with normalized endpoint paths |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint` | End-to-end request latency distribution (13 buckets: 5ms to 10s) |
| `http_requests_in_progress` | Gauge | `method`, `endpoint` | Number of currently concurrent active HTTP requests |
| `http_errors_total` | Counter | `method`, `endpoint`, `status_code` | Count of 4xx and 5xx client and server errors |
| `api_uptime_seconds` | Gauge | None | Total continuous uptime of the FastAPI serving process |
| `model_predictions_total` | Counter | `model_name`, `model_version` | Total model inference operations performed |
| `model_predictions_by_class_total` | Counter | `model_name`, `model_version`, `predicted_class` | Inferences partitioned by classification outcome (`Churn` vs `No Churn`) |
| `model_prediction_probability` | Histogram | `model_name`, `model_version` | Output positive churn probability distribution (calibrated bins including 0.571 cutoff) |
| `model_prediction_errors_total` | Counter | `error_type` | Count of prediction and pipeline execution failures |
| `model_inference_duration_seconds` | Histogram | `model_name`, `model_version` | Sub-millisecond timing of Scikit-learn/XGBoost `predict_proba` (10 buckets: 1ms to 1s) |
| `model_reloads_total` | Counter | `status`, `alias` | Model reload operations grouped by outcome (`success`/`failure`) and alias |
| `model_info` | Gauge | `model_name`, `model_version`, `alias`, `decision_threshold` | Informational gauge (value 1) reflecting currently active champion metadata |
| `model_available` | Gauge | `model_name` | Model readiness status (1 = healthy and loaded, 0 = unavailable) |
| `model_last_reload_timestamp_seconds` | Gauge | `model_name` | Epoch timestamp of the last successful model reload |
| `model_load_failures_total` | Counter | `model_name`, `alias` | Cumulative count of failed model loading or reloading operations |
| `monitoring_runs_total` | Counter | `status` | Total Evidently AI data drift and quality monitoring cycles |
| `monitoring_run_duration_seconds` | Histogram | None | Execution duration of monitoring cycle (7 buckets: 0.5s to 60s) |
| `monitoring_drift_detections_total` | Counter | None | Total confirmed dataset drift events (drift share >= 30%) |
| `monitoring_drift_share` | Gauge | None | Ratio of drifted features over evaluated feature pool in latest cycle |
| `monitoring_data_quality_failures_total` | Counter | None | Cumulative data quality check failures during monitoring runs |
| `monitoring_prediction_drift_detections_total` | Counter | None | Cumulative prediction distribution drift events |
| `monitoring_retraining_recommendations_total` | Counter | `priority`, `trigger_reason` | Retraining recommendations generated (cooldown enforced) |
| `monitoring_ground_truth_feedback_total` | Counter | `source` | Total delayed ground-truth feedback observations received |
| `monitoring_model_roc_auc` | Gauge | None | Supervised ROC-AUC evaluated on matched ground truth |
| `monitoring_model_pr_auc` | Gauge | None | Supervised PR-AUC evaluated on matched ground truth |
| `monitoring_model_f1` | Gauge | None | Supervised F1 score evaluated on matched ground truth |
| `monitoring_model_brier_score` | Gauge | None | Supervised Brier score evaluated on matched ground truth |
| `process_cpu_seconds_total` | Counter | None | Process CPU user and system execution time (via `psutil`) |
| `process_resident_memory_bytes` | Gauge | None | Resident set size (RSS) memory in bytes |
| `process_virtual_memory_bytes` | Gauge | None | Virtual memory size (VMS) in bytes |

### 2. Prometheus Alert Rules Architecture

Alert rules are defined in `monitoring/prometheus/alerts.yml` across three standardized severity tiers. **Governance Invariant**: Alert rules notify operators and engineers; they **never trigger automated model retraining or champion promotion**.

| Severity Tier | Alert Name | Condition (PromQL) | Duration | Description & Operator Playbook |
| :--- | :--- | :--- | :--- | :--- |
| **Critical** | `ApiServiceUnavailable` | `up{job="fastapi"} == 0` | 1m | FastAPI serving container unreachable; restart container and inspect logs |
| **Critical** | `ModelUnavailable` | `model_available == 0` | 1m | Serving engine loaded with no active champion model; check MLflow registry |
| **Critical** | `ElevatedHttp5xxErrors` | `rate(http_errors_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05` | 2m | HTTP 5xx error rate exceeds 5%; inspect FastAPI logs and database connection |
| **Critical** | `ModelReloadFailure` | `increase(model_reloads_total{status="failure"}[5m]) > 0` | 0m | Atomic model reload failed; previous healthy champion was retained |
| **Warning** | `HighApiRequestLatency` | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) > 0.5` | 5m | p95 request latency exceeds 500ms; check CPU utilization and database pool |
| **Warning** | `HighModelInferenceLatency` | `histogram_quantile(0.95, sum(rate(model_inference_duration_seconds_bucket[5m])) by (le)) > 0.03` | 5m | p95 inference time exceeds 30ms target; check model pipeline and thread contention |
| **Warning** | `ElevatedPredictionErrors` | `rate(model_prediction_errors_total[5m]) > 0.01` | 3m | Elevated inference failure rate; inspect input schemas and pipeline transforms |
| **Warning** | `DatasetDriftDetected` | `monitoring_drift_share >= 0.30` | 0m | Feature drift share >= 30%; feature distribution has diverged from reference |
| **Warning** | `DataQualityFailures` | `increase(monitoring_data_quality_failures_total[1h]) > 0` | 0m | Input data violated quality contracts (missing values, novel categories) |
| **Warning** | `PredictionDriftDetected` | `increase(monitoring_prediction_drift_detections_total[1h]) > 0` | 0m | Churn prediction distribution diverged significantly from baseline |
| **Warning** | `MonitoringCycleFailed` | `increase(monitoring_runs_total{status="failure"}[1h]) > 0` | 0m | Evidently AI background monitoring cycle crashed; inspect worker logs |
| **Informational** | `RetrainingRecommendationGenerated` | `increase(monitoring_retraining_recommendations_total[1h]) > 0` | 0m | Retraining recommendation created; review evaluation reports before manual trigger |

### 3. Auto-Provisioned Grafana Dashboards

Grafana is provisioned automatically with Prometheus datasource and four pre-configured operational dashboards:

1. **Dashboard A: API Overview (`monitoring/grafana/dashboards/api_overview.json`)**:
   - Status header: API availability status and current uptime.
   - Traffic rates: Total HTTP request rate (req/s) broken down by normalized endpoint.
   - Error tracking: HTTP 5xx and 4xx rate and error ratio percentage.
   - Status code distribution: Donut chart of 2xx, 4xx, and 5xx responses.
   - Latency distributions: Multi-series panel tracking p50, p90, p95, and p99 request latencies.
2. **Dashboard B: ML Inference & Model Performance (`monitoring/grafana/dashboards/ml_inference.json`)**:
   - Model info panel: Current champion model name, active version, alias, and calibrated decision threshold (0.571).
   - Prediction throughput: Inference requests per second partitioned by predicted class (`Churn` vs `No Churn`).
   - Probability distribution: Real-time histogram of output churn probabilities with vertical threshold marker.
   - Inference latency: High-precision Scikit-learn/XGBoost `predict_proba` p50, p95, and p99 percentiles (target: < 30ms).
   - Reload events: History and status of model reload operations.
3. **Dashboard C: Model Monitoring & Data Drift (`monitoring/grafana/dashboards/model_monitoring.json`)**:
   - Drift share gauge: Real-time gauge of current drifted feature share with color-coded warning (>= 30%) and critical (>= 50%) thresholds.
   - Cumulative drift events: Step chart of confirmed dataset drift detections.
   - Data quality failures: Hourly failure count of missing values and schema anomalies.
   - Retraining recommendations: Timeline of retraining alerts grouped by priority and root-cause trigger.
   - Supervised performance: Ground-truth feedback tracking with real-time ROC-AUC, PR-AUC, F1-Score, and Brier calibration curves.
4. **Dashboard D: Infrastructure & System Resources (`monitoring/grafana/dashboards/infrastructure.json`)**:
   - Target scrape health: Prometheus scrape status for FastAPI and internal Prometheus instances.
   - Memory consumption: Process resident memory (RSS) and virtual memory (VMS) usage in megabytes.
   - CPU utilization: Percentage process CPU consumption derived from `process_cpu_seconds_total`.
   - Monitoring execution: Duration of Evidently AI background monitoring evaluation cycles.

### 4. Docker Compose Observability Topology

Prometheus and Grafana are seamlessly integrated into `docker-compose.yml`:

```bash
# Start complete containerized stack including Prometheus and Grafana
docker compose up -d

# Check running observability services
docker compose ps
# mlops-api         :8000
# mlops-mlflow      :5000
# mlops-postgres    :5432
# mlops-prometheus  :9090
# mlops-grafana     :3000
```

- **Prometheus URL**: `http://localhost:9090` (Scrapes FastAPI `/metrics` every 5 seconds).
- **Grafana URL**: `http://localhost:3000` (Default credentials: `admin` / `admin`).


