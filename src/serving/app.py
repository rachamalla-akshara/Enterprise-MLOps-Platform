"""FastAPI Application Factory for Model Serving Engine."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import time
from typing import Optional, Union
import uuid
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.common.config import AppConfig, get_config
from src.common.logger import get_logger
from src.monitoring.monitoring_service import MonitoringService
from src.observability import (
    PrometheusMetricsMiddleware,
    get_latest_metrics,
    record_model_loaded,
    record_model_reload_event,
    record_model_unavailable,
    record_prediction_error,
)
from src.serving.exceptions import (
    AuthenticationError,
    ModelNotLoadedError,
    ModelReloadError,
    PredictionError,
    ServingError,
)
from src.serving.model_loader import ModelLoader
from src.serving.prediction_logger import PredictionLogger
from src.serving.prediction_service import PredictionService
from src.serving.schemas import (
    CustomerFeatures,
    GroundTruthFeedbackRequest,
    GroundTruthFeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringRunResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
    ReloadRequest,
    ReloadResponse,
)

logger = get_logger(__name__)


def create_app(
    config: Optional[AppConfig] = None,
    model_loader: Optional[ModelLoader] = None,
    prediction_logger: Optional[PredictionLogger] = None,
    monitoring_service: Optional[MonitoringService] = None,
) -> FastAPI:
    """Creates and configures a FastAPI application instance.

    Args:
        config: Application configuration.
        model_loader: Optional custom/mocked ModelLoader instance.
        prediction_logger: Optional custom/mocked PredictionLogger instance.
        monitoring_service: Optional custom/mocked MonitoringService instance.

    Returns:
        FastAPI: Configured serving application.
    """
    cfg = config or get_config()
    loader = model_loader or ModelLoader(cfg)
    logger_db = prediction_logger or PredictionLogger(config=cfg)
    prediction_svc = PredictionService(loader, logger_db, cfg)
    monitoring_svc = monitoring_service or MonitoringService(app_config=cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifecycle manager for startup initialization and shutdown teardown."""
        logger.info("Initializing FastAPI Serving Application: %s", cfg.project.name)
        try:
            loader.load_champion_model()
            logger.info("Model Serving Engine successfully loaded active champion model.")
            container = loader.get_active_model()
            record_model_loaded(
                model_name=container.model_name,
                model_version=container.model_version,
                alias=container.alias,
                decision_threshold=float(cfg.serving.decision_threshold),
            )
        except Exception as e:
            logger.error(
                "CRITICAL: Failed to load champion model on startup: %s. Serving engine will start in unready state.",
                e,
            )
            record_model_unavailable(cfg.mlflow.registered_model_name)

        yield

        logger.info("Shutting down FastAPI Serving Engine. Disposing database resources...")
        try:
            if hasattr(logger_db, "engine") and hasattr(logger_db.engine, "dispose"):
                logger_db.engine.dispose()
            if hasattr(monitoring_svc, "engine") and hasattr(monitoring_svc.engine, "dispose"):
                monitoring_svc.engine.dispose()
        except Exception as e:
            logger.warning("Error disposing database engine on shutdown: %s", e)

    app = FastAPI(
        title=f"{cfg.project.name} - Model Serving Engine",
        version=cfg.project.version,
        description="High-performance, dynamic model serving engine integrated with MLflow Model Registry.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Attach instances to app state
    app.state.config = cfg
    app.state.model_loader = loader
    app.state.prediction_logger = logger_db
    app.state.prediction_service = prediction_svc
    app.state.monitoring_service = monitoring_svc

    # Attach Prometheus metrics middleware
    app.add_middleware(PrometheusMetricsMiddleware)

    # --------------------------------------------------------------------------
    # Middleware: Request ID & Request Latency Logging
    # --------------------------------------------------------------------------
    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id

        t0 = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        # Log requests except health probes to reduce log noise
        if request.url.path not in ["/health", "/ready"]:
            logger.info(
                "[%s] %s %s - Status: %d (%0.2f ms)",
                req_id,
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        return response

    # --------------------------------------------------------------------------
    # Global Exception Handlers
    # --------------------------------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.warning("[%s] Request validation failed: %s", req_id, exc.errors())
        record_prediction_error(error_type="validation_error")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "InputValidationError",
                "message": "Input features failed validation constraints.",
                "request_id": req_id,
                "details": exc.errors(),
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error("[%s] Model not loaded: %s", req_id, exc)
        record_prediction_error(error_type="model_not_loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "ModelNotLoadedError",
                "message": str(exc.message),
                "request_id": req_id,
                "details": exc.details,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(ModelReloadError)
    async def model_reload_handler(request: Request, exc: ModelReloadError):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error("[%s] Model reload failed: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "ModelReloadError",
                "message": str(exc.message),
                "request_id": req_id,
                "details": exc.details,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(request: Request, exc: AuthenticationError):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "AuthenticationError",
                "message": str(exc.message),
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(request: Request, exc: PredictionError):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.error("[%s] Prediction error: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "PredictionError",
                "message": str(exc.message),
                "request_id": req_id,
                "details": exc.details,
            },
            headers={"X-Request-ID": req_id},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger.exception("[%s] Unhandled server error: %s", req_id, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred during request processing.",
                "request_id": req_id,
            },
            headers={"X-Request-ID": req_id},
        )

    # --------------------------------------------------------------------------
    # Security Dependency for Protected Endpoints
    # --------------------------------------------------------------------------
    def verify_admin_token(
        x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-API-Key"),
        authorization: Optional[str] = Header(None),
    ) -> None:
        """Verifies admin authorization token for protected administrative operations."""
        if not cfg.serving.enable_reload_endpoint:
            raise AuthenticationError("Administrative reload endpoint is disabled by configuration.")

        expected = cfg.serving.reload_token

        # Check X-Admin-API-Key header
        if x_admin_api_key and x_admin_api_key == expected:
            return

        # Check Bearer authorization header
        if authorization and authorization.startswith("Bearer "):
            token = authorization.replace("Bearer ", "").strip()
            if token == expected:
                return

        raise AuthenticationError("Invalid or missing administrative authorization credentials.")

    # --------------------------------------------------------------------------
    # API Endpoints
    # --------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Process health probe verifying FastAPI runtime responsiveness."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/ready", response_model=ReadyResponse, tags=["Health"])
    async def readiness_check():
        """Readiness probe verifying that an active champion model is loaded."""
        if not loader.is_ready():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "model_loaded": False,
                    "message": "Model is not loaded or registry connection is unavailable.",
                    "ready_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        container = loader.get_active_model()
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            model_name=container.model_name,
            model_version=container.model_version,
            model_alias=container.alias,
            ready_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Governance"])
    async def model_info():
        """Returns metadata for the currently active registered model."""
        container = loader.get_active_model()
        return ModelInfoResponse(
            model_name=container.model_name,
            resolved_version=container.model_version,
            run_id=container.run_id,
            alias=container.alias,
            load_timestamp=container.load_timestamp.isoformat(),
            decision_threshold=float(cfg.serving.decision_threshold),
            status="active",
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
    async def predict_churn(
        payload: Union[CustomerFeatures, PredictionRequest],
        request: Request,
    ):
        """Generates churn prediction and probability for a validated customer record.

        Accepts either flat customer feature attributes or a wrapped PredictionRequest.
        Strictly forbids the target column 'Churn'.
        """
        req_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Extract features whether passed flat or wrapped
        if isinstance(payload, PredictionRequest):
            features = payload.features
        else:
            features = payload

        # Optional periodic alias check if configured
        if cfg.serving.auto_reload_interval_seconds > 0:
            loader.check_and_update_if_alias_changed()

        response = prediction_svc.predict(features=features, request_id=req_id)
        return response

    @app.post(
        "/model/reload",
        response_model=ReloadResponse,
        dependencies=[Depends(verify_admin_token)],
        tags=["Model Governance"],
    )
    async def reload_model(reload_req: Optional[ReloadRequest] = None):
        """Safely reloads model from MLflow registry and atomically updates the active pipeline.

        Requires administrative authorization via 'X-Admin-API-Key' or 'Authorization: Bearer <token>'.
        If candidate model verification fails, the existing active model remains untouched.
        """
        alias = reload_req.alias if reload_req else None
        target_alias = alias or cfg.serving.model_alias

        try:
            old_container, new_container = loader.reload(alias=target_alias)
            model_name = getattr(new_container, "model_name", cfg.mlflow.registered_model_name)
            record_model_reload_event(status="success", alias=target_alias, model_name=model_name)
            record_model_loaded(
                model_name=model_name,
                model_version=getattr(new_container, "model_version", "unknown"),
                alias=target_alias,
                decision_threshold=float(cfg.serving.decision_threshold),
            )
        except Exception:
            model_name = getattr(loader, "model_name", None) or cfg.mlflow.registered_model_name
            record_model_reload_event(status="failure", alias=target_alias, model_name=model_name)
            raise

        return ReloadResponse(
            status="reloaded",
            message=f"Model successfully reloaded for alias '{target_alias}'.",
            previous_version=old_container.model_version if old_container else "None",
            new_version=new_container.model_version,
            model_name=new_container.model_name,
            reloaded_at=datetime.now(timezone.utc).isoformat(),
        )

    # --------------------------------------------------------------------------
    # Monitoring Endpoints
    # --------------------------------------------------------------------------
    @app.get("/monitoring/status", tags=["Model Monitoring"])
    async def monitoring_status():
        """Returns current model monitoring status, latest drift evaluation, and recommendations."""
        return monitoring_svc.get_latest_status()

    @app.get("/monitoring/latest", tags=["Model Monitoring"])
    async def monitoring_latest_report():
        """Returns the most recent dataset drift audit report."""
        status_data = monitoring_svc.get_latest_status()
        latest_drift = status_data.get("latest_drift_report")
        if not latest_drift:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No monitoring reports have been generated yet.",
            )
        return latest_drift

    @app.post(
        "/monitoring/run",
        response_model=MonitoringRunResponse,
        dependencies=[Depends(verify_admin_token)],
        tags=["Model Monitoring"],
    )
    async def run_monitoring_cycle():
        """Triggers an on-demand statistical monitoring and drift detection cycle.

        Requires administrative authorization via 'X-Admin-API-Key' or 'Authorization: Bearer <token>'.
        """
        summary = monitoring_svc.run_monitoring_cycle()
        return MonitoringRunResponse(
            status="completed",
            report_id=summary.report_id,
            dataset_drift_detected=summary.dataset_drift_detected,
            drift_share=summary.drift_share,
            data_quality_status=summary.data_quality_status,
            prediction_drift_detected=summary.prediction_drift_detected,
            retraining_recommendation_status=summary.retraining_recommendation_status,
            retraining_trigger_reason=summary.retraining_trigger_reason,
            report_html_path=summary.report_html_path,
            report_json_path=summary.report_json_path,
            executed_at=summary.timestamp,
        )

    @app.post(
        "/monitoring/feedback",
        response_model=GroundTruthFeedbackResponse,
        tags=["Model Monitoring"],
    )
    async def submit_ground_truth_feedback(feedback: GroundTruthFeedbackRequest):
        """Ingests delayed ground truth churn outcome for a prediction request."""
        monitoring_svc.ingest_ground_truth(
            request_id=feedback.request_id,
            actual_churn=feedback.actual_churn,
            source=feedback.source,
        )
        return GroundTruthFeedbackResponse(
            status="success",
            request_id=feedback.request_id,
            actual_churn=feedback.actual_churn,
            message="Ground truth feedback recorded successfully.",
        )

    # --------------------------------------------------------------------------
    # Observability & Metrics Endpoints
    # --------------------------------------------------------------------------
    @app.get(
        "/metrics",
        tags=["Observability"],
        summary="Prometheus metrics exposition endpoint",
        response_class=Response,
    )
    async def prometheus_metrics():
        """Exposes application, inference, model governance, and monitoring metrics in Prometheus format."""
        content, media_type = get_latest_metrics()
        return Response(content=content, media_type=media_type)

    return app


# Default application instance for uvicorn entrypoint (e.g. uvicorn src.serving.app:app)
app = create_app()
