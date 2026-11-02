"""Prediction Telemetry and Audit Logger using SQLAlchemy."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.common.config import AppConfig, find_project_root, get_config
from src.common.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class PredictionLogRecord(Base):
    """SQLAlchemy model for recording inference requests and model predictions.

    Does not store sensitive personal customer data or raw feature payloads.
    Fully compatible with SQLite for local development and PostgreSQL for production.
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String(128), nullable=False)
    model_version = Column(String(32), nullable=False)
    prediction = Column(String(32), nullable=False)
    churn_probability = Column(Float, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    inference_latency_ms = Column(Float, nullable=False)
    request_status = Column(String(32), nullable=False, default="SUCCESS")

    def to_dict(self) -> Dict[str, Any]:
        """Converts model row to dictionary."""
        return {
            "id": self.id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "prediction": self.prediction,
            "churn_probability": self.churn_probability,
            "decision_threshold": self.decision_threshold,
            "inference_latency_ms": self.inference_latency_ms,
            "request_status": self.request_status,
        }


class PredictionLogger:
    """Telemetry logger persisting prediction audit logs to local SQLite or PostgreSQL.

    Guarantees:
    - Zero sensitive personal identifiable information stored.
    - Database logging failures are logged but never fail a successful client prediction response.
    """

    def __init__(self, db_uri: Optional[str] = None, config: Optional[AppConfig] = None) -> None:
        self.cfg = config or get_config()
        self.root = find_project_root()

        uri = db_uri or self.cfg.serving.prediction_db_uri
        if uri.startswith("sqlite:///"):
            db_rel = uri.replace("sqlite:///", "")
            db_abs = (self.root / db_rel).resolve()
            self.db_uri = f"sqlite:///{db_abs.as_posix()}"
        else:
            self.db_uri = uri

        self.engine = create_engine(
            self.db_uri,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_uri else {},
            pool_pre_ping=True,
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()

    def _init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Initialized prediction telemetry database at: %s", self.db_uri)
        except Exception as e:
            logger.error("Failed to initialize prediction database '%s': %s", self.db_uri, e)

    def log_prediction(
        self,
        request_id: str,
        model_name: str,
        model_version: str,
        prediction: str,
        churn_probability: float,
        decision_threshold: float,
        inference_latency_ms: float,
        request_status: str = "SUCCESS",
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Safely persists prediction audit record.

        Failure in logging logs a warning but DOES NOT raise an exception to the caller.

        Returns:
            bool: True if persisted successfully, False if logging failed.
        """
        record = PredictionLogRecord(
            request_id=request_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            model_name=model_name,
            model_version=str(model_version),
            prediction=prediction,
            churn_probability=float(churn_probability),
            decision_threshold=float(decision_threshold),
            inference_latency_ms=float(inference_latency_ms),
            request_status=request_status,
        )

        try:
            with self.SessionLocal() as session:
                session.add(record)
                session.commit()
            return True
        except Exception as e:
            logger.error(
                "CRITICAL: Failed to write prediction log to database: request_id=%s, error=%s",
                request_id,
                e,
            )
            return False

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves most recent prediction logs for verification and monitoring."""
        try:
            with self.SessionLocal() as session:
                stmt = select(PredictionLogRecord).order_by(PredictionLogRecord.id.desc()).limit(limit)
                records = session.scalars(stmt).all()
                return [r.to_dict() for r in records]
        except Exception as e:
            logger.error("Failed to query prediction logs: %s", e)
            return []
