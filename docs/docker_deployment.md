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

### 2.3 FastAPI Worker Concurrency & In-Memory Model Lifecycle Consistency
The containerized FastAPI serving engine is explicitly configured to run with **exactly one worker** (`--workers 1`):
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Architectural Rationale**:
- **In-Memory State**: The `ModelLoader` dynamically maintains the active scikit-learn pipeline and decision threshold in process memory.
- **Dynamic Reload Isolation**: When an administrator issues `POST /model/reload`, the reload logic updates the in-memory state of the process handling that specific HTTP request. In a multi-worker Uvicorn deployment without an inter-process communication (IPC) broadcast bus (e.g. Redis Pub/Sub), only one worker process would reload the model, causing **split-brain divergence** where different worker processes serve different model versions simultaneously.
- **Horizontal Scaling Paradigm**: High-throughput concurrency is achieved by scaling stateless container replicas horizontally across Kubernetes pods or Docker Compose replicas behind a load balancer, rather than multiplexing worker processes within a single container. Multi-worker reload consistency is deliberately not claimed until distributed synchronization is introduced.

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

#### Dual-URI & Integrity Enforcement:
1. **Source & Target Isolation**: Strictly forbids migrating when source and target URIs are identical without `--audit-only`.
2. **Network Probe & Offline Detection**: Probes target database connectivity with an immediate network timeout. If the destination PostgreSQL container is stopped, the tool records audit status `PENDING_TARGET_BACKEND` with distinct source and target URIs and preserves the intact SQLite database.
3. **Full Run & Parameter Migration**: Iterates through source runs, creating runs in the destination experiment and copying all logged parameters, evaluation metrics, and run tags.
4. **Physical Artifact Transfer**: Copies model artifacts (`mlruns/models/<id>/artifacts`) to the configured destination artifact store and updates model version source pointers.
5. **Alias Re-establishment**: Binds `@champion`, `@challenger`, and `@candidate` aliases to the migrated target versions.
6. **Prediction Parity Assertion**: Loads `@champion` from both source and target registries, executes inference on real customer sample records, and strictly enforces numerical equality via `np.testing.assert_allclose(atol=1e-5)`.
7. **Audit Record**: Emits JSON verification record to `reports/evaluations/mlflow_migration_audit.json`.

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

## 7. Phase 4.2 Comprehensive Live Deployment Verification & Audit Matrix

The Phase 4.2 live deployment verification has been completed end-to-end against active Docker containers, validating multi-stage container builds, PostgreSQL database initialization, MLflow SQLite-to-PostgreSQL migration, real model dynamic resolution, live REST inference, authenticated dynamic reload, zero-PII telemetry logging, and named volume persistence across container restarts.

| Item | Verification Category | Status | Operational Evidence & Verification Details |
| :---: | :--- | :---: | :--- |
| **A** | **Docker Engine & CLI Verification** | **PASS** | Docker Engine `29.1.3` (WSL2 Kernel `6.6.87.2-microsoft-standard-WSL2`) and Docker Compose `v5.0.0-desktop.1` verified operational. |
| **B** | **Configuration & Credential Harmonization** | **PASS** | Harmonized PostgreSQL credentials across `docker-compose.yml`, `.env.example`, `scripts/init_postgres.sql`, and `scripts/migrate_mlflow_registry.py` with standard `mlops_user:mlops_password@postgres:5432/mlops_db`. |
| **C** | **Multi-Stage Production Docker Build** | **PASS** | Built `enterprise-mlops-api:1.0.0` from `Dockerfile` (two stages: `builder` and unprivileged `runner`). Content image size: **745 MB**. Verified runtime non-root execution (`uid=10001(mlops) gid=10001(mlops)`). |
| **D** | **Compose Multi-Container Startup & Health** | **PASS** | `mlops-postgres`, `mlops-mlflow`, and `mlops-api` launched and reported `healthy`. Healthchecks: `pg_isready -U mlops_user -d mlops_db` for Postgres; `curl -f http://localhost:5000/health` for MLflow; `curl -f http://localhost:8000/health` for FastAPI. |
| **E** | **MLflow PostgreSQL Migration & Parity** | **PASS** | Migrated 4 runs, all metrics/parameters/tags, and all 4 model versions of `customer-churn-classifier` from `sqlite:///mlflow.db` into PostgreSQL. Registered model aliases established (`@champion` $\to$ v4, `@challenger` $\to$ v3, `@candidate` $\to$ v4). Prediction parity verified with `np.testing.assert_allclose(atol=1e-5)` against holdout test records. Migration audit saved to `reports/evaluations/mlflow_migration_audit.json` with status `SUCCESS`. |
| **F** | **FastAPI Dynamic Champion Model Loading** | **PASS** | `GET http://localhost:8000/ready` returned `200 OK` with model `customer-churn-classifier` version `4` under alias `champion`. `GET http://localhost:8000/model/info` returned `200 OK` with run ID `a5f011c5e93f423799ea0124f7611385`, threshold `0.571`, and status `active`. |
| **G** | **Live Inference Execution in Container** | **PASS** | Executed `POST http://localhost:8000/predict` on live Telco customer sample. HTTP `200 OK` returned: prediction `Churn`, probability `0.7403` (and post-restart sample probability `0.5781`), latency `35.72 ms` (post-restart `20.90 ms`). |
| **H** | **Swagger UI & OpenAPI Specification** | **PASS** | `GET http://localhost:8000/docs` returned HTTP `200 OK` with interactive OpenAPI documentation for all serving endpoints. |
| **I** | **Dynamic Reload & Security Authentication** | **PASS** | `POST http://localhost:8000/model/reload` with `X-Admin-API-Key: dev-admin-secret-key-phase3` returned HTTP `200 OK` (`{"status": "reloaded", "previous_version": "4", "new_version": "4"}`). Unauthorized requests rejected with `401 Unauthorized`. |
| **J** | **PostgreSQL Telemetry Persistence (Zero PII)** | **PASS** | Queried `public.prediction_logs` in PostgreSQL: confirmed records persisted with `request_id`, `model_name`, `model_version`, `prediction`, `churn_probability`, `decision_threshold`, `inference_latency_ms`, and `timestamp`. Customer feature inputs are excluded, enforcing zero-PII compliance. |
| **K** | **Container Restart & Volume Persistence** | **PASS** | Executed `docker compose restart`. All 3 services resumed and returned to `Up (healthy)`. Active model version 4 remained loaded without re-downloading. Verified both pre- and post-restart prediction logs (ID 1 and 2) preserved in PostgreSQL. |
| **L** | **Regression Test Suite** | **PASS** | Complete regression test suite passes (**51 passed, 0 failed**) covering data ingestion, ML pipelines, MLflow registry governance, model loader, prediction service, telemetry database logging, and migration tool. |
