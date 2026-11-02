# ==============================================================================
# Enterprise MLOps Platform — Multi-Stage Production Dockerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependencies Compiler
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies required for compiling C extensions (psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install application dependencies reproducibly
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runner

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Install minimal runtime shared libraries (libpq for postgres, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create dedicated non-root application user and group (UID/GID 10001)
RUN groupadd --gid 10001 mlops && \
    useradd --uid 10001 --gid 10001 --shell /bin/bash --create-home mlops

# Set working directory and establish ownership
WORKDIR /app
RUN chown -R mlops:mlops /app

# Copy application configuration and codebase
COPY --chown=mlops:mlops configs/ ./configs/
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops data/ ./data/
COPY --chown=mlops:mlops models/ ./models/
COPY --chown=mlops:mlops reports/ ./reports/

# Switch to unprivileged non-root user
USER 10001:10001

# Healthcheck verifying FastAPI application liveness
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose FastAPI serving port
EXPOSE 8000

# Reliable production entrypoint
# Note on worker concurrency: A single worker (--workers 1) is strictly required for in-memory model lifecycle
# management. Dynamic reload operations (/model/reload) update the ModelLoader state within process memory.
# In a multi-worker setup without an IPC broadcast mechanism (e.g., Redis Pub/Sub), a reload request would
# only update the worker that handled the HTTP request, leading to split-brain model divergence across workers.
# Horizontal scaling is achieved across container replicas behind a load balancer, not via multi-worker Uvicorn.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
