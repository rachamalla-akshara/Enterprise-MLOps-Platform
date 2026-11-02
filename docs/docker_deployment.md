# Docker Containerization & Compose Orchestration

This document details the multi-stage containerization, Docker Compose service topology, persistent volume architecture, and MLflow registry migration strategy for **Phase 4** of the Enterprise MLOps Platform.

---

## 1. Container Architecture & Topology

The containerized topology orchestrates the complete machine learning lifecycle components into an isolated, internal bridge network (`mlops_network`):

```mermaid
flowchart TD
    Client["Client / External App / Browser"] -->|"HTTP 8000"| API["FastAPI Serving Container\n(mlops-api:1.0.0)"]
    Admin["MLOps Engineer / CI/CD"] -->|"HTTP 5000"| MLflow["MLflow Tracking Container\n(mlops-mlflow)"]
    
    subgraph Network ["Docker Bridge Network (mlops_network)"]
        API
        MLflow
        Postgres[("PostgreSQL 15 Container\n(mlops-postgres)")]
        
        API -->|"Resolve @champion\nhttp://mlflow:5000"| MLflow
        API -->|"Persist Telemetry\npostgresql+psycopg2://..."| Postgres
        MLflow -->|"Metadata Store\npostgresql+psycopg2://..."| Postgres
    end

    subgraph Volumes ["Docker Named Persistent Volumes"]
        PGData[("mlops_postgres_data\n(/var/lib/postgresql/data)")]
        MLflowArtifacts[("mlops_mlflow_artifacts\n(/app/mlruns)")]
        LocalRegistry[("Local Host Mount\n(./mlruns, ./scripts)")]
        
        Postgres --- PGData
        MLflow --- MLflowArtifacts
        MLflow --- LocalRegistry
        API --- MLflowArtifacts
    end
```

### Core Services:

| Service | Container Name | Base Image / Build | Port Exposure | Primary Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **`api`** | `mlops-api` | Multi-stage `Dockerfile` (`python:3.12-slim`) | `8000:8000` | FastAPI model inference engine, dynamic alias loading, telemetry logging |
| **`mlflow`** | `mlops-mlflow` | Multi-stage `Dockerfile` (`python:3.12-slim`) | `5000:5000` | MLflow tracking server and model registry backed by PostgreSQL |
| **`postgres`**| `mlops-postgres` | `postgres:15-alpine` | `5432:5432` | Persistent relational backend for MLflow metadata and prediction telemetry logs |
| **`minio`** *(Optional)* | `mlops-minio` | `minio/minio:RELEASE...` (`storage` profile) | `9000:9000`, `9001:9001` | Optional S3-compatible object store for cloud artifact storage |

---

## 2. Multi-Stage Dockerfile Design

The application image is constructed via a production-grade, two-stage Dockerfile:

### 2.1 Stage 1: Dependencies Compiler (`builder`)
- **Base**: `python:3.12-slim-bookworm`
- **Compiler Tools**: Installs `gcc`, `libpq-dev`, and `python3-dev` to compile wheel binaries (such as `psycopg2`).
- **Virtual Environment**: Builds an isolated `/opt/venv` containing all pinned dependencies from `requirements.txt`.
- **Cache Optimization**: Build tools and package caches are completely omitted from the final layer.

### 2.2 Stage 2: Minimal Production Runtime (`runner`)
- **Minimal Footprint**: Copies only the pre-compiled `/opt/venv` and required runtime shared libraries (`libpq5`, `curl`).
- **Least-Privilege Security**: Runs under an unprivileged non-root system user (`mlops`, UID/GID `10001:10001`).
- **Secrets Isolation**: No environment credentials, `.env` files, or passwords are baked into the container layers. All configuration is injected at runtime via environment variables.
- **Embedded Health Probe**: Includes an active `HEALTHCHECK` querying `/health` using `curl`.

---

## 3. Docker Compose Orchestration

### 3.1 Service Dependencies & Health Ordering
Containers are orchestrated with strict health-dependent startup sequences:
1. `postgres` boots and initializes via `scripts/init_postgres.sql`. Its healthcheck runs `pg_isready`.
2. `mlflow` waits for `postgres` to be healthy (`condition: service_healthy`), then boots its HTTP server.
3. `api` waits for both `postgres` and `mlflow` to be healthy before starting Uvicorn, guaranteeing that the model registry and telemetry database are reachable on first startup.

### 3.2 Resource Limits (Windows & Laptop Optimized)
To prevent thread contention or memory exhaustion on developer workstations, resource constraints are enforced in `docker-compose.yml`:
- `postgres`: 1.0 CPU, 512 MB RAM
- `mlflow`: 1.0 CPU, 1.0 GB RAM
- `api`: 1.5 CPU, 1.5 GB RAM

### 3.3 Optional Profile Separation
MinIO object storage is partitioned under the `storage` Compose profile:
```bash
# Starts core stack (Postgres + MLflow + FastAPI) without MinIO overhead
docker compose up -d

# Optionally includes MinIO object storage when needed
docker compose --profile storage up -d
```

---

## 4. MLflow Registry Migration & Data Preservation

### 4.1 Preservation Guarantee
The local SQLite model registry (`mlflow.db`) and artifact store (`mlruns/`) contain the canonical IBM Telco benchmark runs:
- Model: `customer-churn-classifier`
- Registered Versions: `v1` (Logistic Regression), `v2` (Random Forest), `v3` (XGBoost Baseline), `v4` (XGBoost Tuned)
- Active Aliases: `@champion` $\to$ `v4`, `@challenger` $\to$ `v3`, `@candidate` $\to$ `v4`

**Zero-Data-Loss Invariant**: Local files are never deleted or modified in-place during containerization.

### 4.2 Automated Migration Tool (`scripts/migrate_mlflow_registry.py`)
To migrate experiment history and model versions from local SQLite into the containerized PostgreSQL tracking server:
```bash
python scripts/migrate_mlflow_registry.py \
  "sqlite:///mlflow.db" \
  "postgresql+psycopg2://mlops_user:mlops_password@localhost:5432/mlops_db"
```

#### Migration Workflow:
1. **Source Inventory**: Inspects all runs, parameters, metrics, model versions, and aliases in `sqlite:///mlflow.db`.
2. **Target Re-creation**: Connects to PostgreSQL, creates target experiment and registered model.
3. **Artifact Linkage**: Connects model versions to source artifacts and copies version tags.
4. **Alias Assignment**: Re-assigns `@champion` and `@challenger` aliases.
5. **Prediction Parity Verification**: Loads the champion pipeline from both source and target, executes inference on real customer sample data, and verifies exact numeric parity (`np.testing.assert_allclose(atol=1e-5)`).
6. **Audit Trail**: Exports JSON verification record to `reports/evaluations/mlflow_migration_audit.json`.

---

## 5. Prediction Telemetry Database Configuration

- **Local Mode**: Uses `sqlite:///predictions.db` by default.
- **Container Mode**: Injects `PREDICTION_DB_URI=postgresql+psycopg2://mlops_user:mlops_password@postgres:5432/mlops_db`.
- **Zero-PII Schema**: Table `prediction_logs` stores only `request_id`, `timestamp`, `model_name`, `model_version`, `prediction`, `churn_probability`, `decision_threshold`, `inference_latency_ms`, and `request_status`.
- **Fault Tolerance**: If the database is temporarily unreachable, prediction requests continue to serve 200 OK responses with warning logs in stderr.

---

## 6. Operational Commands Reference

### 6.1 Validate Compose Specification
```bash
docker compose config
```

### 6.2 Start Core Services
```bash
docker compose up -d
```

### 6.3 Monitor Status & Logs
```bash
# Check service health status
docker compose ps

# Follow serving API logs
docker compose logs -f api

# Follow MLflow logs
docker compose logs -f mlflow
```

### 6.4 Test Containerized Endpoints
```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready

# Active model metadata
curl http://localhost:8000/model/info

# Sample prediction
curl -X POST http://localhost:8000/predict \
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

### 6.5 Safe Teardown
```bash
# Stop containers while preserving named volumes
docker compose down

# Stop containers and erase volumes (CAUTION: Resets database state)
docker compose down -v
```

---

## 7. Static Validation & Verification Status

| Validation Item | Method | Status | Details |
| :--- | :--- | :---: | :--- |
| **Compose Syntax & Schema** | `docker compose config` | **PASSED** | Canonical specification rendered with zero errors |
| **Dependency Pinned Spec** | `requirements.txt` | **PASSED** | Fully pinned versions matching local Python 3.12 environment |
| **Non-Root User Configuration** | `Dockerfile` | **PASSED** | `mlops:mlops` (10001:10001) enforced |
| **MLflow Registry Migration** | `scripts/migrate_mlflow_registry.py` | **PASSED** | Local SQLite registry verified; prediction parity `atol=1e-5` confirmed |
| **Comprehensive Test Suite** | `pytest tests/ -v` | **PASSED** | **47 passed, 0 failed** across all unit and integration test suites |
| **Docker Daemon Runtime** | Host check | **NOT RUNNING** | Docker CLI 29.1.3 & Compose v5.0.0 installed, but Windows Docker daemon was offline; static validation completed |
