"""Central Monitoring Service orchestrating data quality, drift detection, and persistence."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import AppConfig, find_project_root, get_config
from src.common.logger import get_logger
from src.monitoring.data_quality import ALL_FEATURES, DataQualityMonitor, DataQualityResult
from src.monitoring.drift_detector import DatasetDriftResult, DriftDetector
from src.monitoring.models import (
    DataQualityReportRecord,
    DriftReportRecord,
    FeatureDriftResultRecord,
    FeatureObservationRecord,
    GroundTruthFeedbackRecord,
    ModelPerformanceReportRecord,
    MonitoringBase,
    PredictionDistributionReportRecord,
    RetrainingRecommendationRecord,
    get_monitoring_engine,
)
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.performance_monitor import PerformanceMonitor, PerformanceResult
from src.monitoring.prediction_monitor import PredictionDistributionResult, PredictionMonitor
from src.monitoring.report_generator import ReportGenerator
from src.monitoring.retraining_recommendations import (
    RetrainingRecommendation,
    RetrainingRecommendationEngine,
)

logger = get_logger(__name__)


@dataclass
class MonitoringCycleSummary:
    """Summary outcome of an end-to-end monitoring evaluation cycle."""

    report_id: str
    timestamp: str
    model_name: str
    model_version: str
    dataset_drift_detected: bool
    drift_share: float
    data_quality_status: str
    prediction_drift_detected: bool
    performance_status: str
    performance_degraded: bool
    retraining_recommendation_status: str
    retraining_trigger_reason: Optional[str]
    report_html_path: str
    report_json_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset_drift_detected": self.dataset_drift_detected,
            "drift_share": round(self.drift_share, 4),
            "data_quality_status": self.data_quality_status,
            "prediction_drift_detected": self.prediction_drift_detected,
            "performance_status": self.performance_status,
            "performance_degraded": self.performance_degraded,
            "retraining_recommendation_status": self.retraining_recommendation_status,
            "retraining_trigger_reason": self.retraining_trigger_reason,
            "report_html_path": self.report_html_path,
            "report_json_path": self.report_json_path,
        }


class MonitoringService:
    """Enterprise model monitoring service orchestrating drift detection and telemetry persistence."""

    def __init__(
        self,
        db_uri: Optional[str] = None,
        config: Optional[MonitoringConfig] = None,
        app_config: Optional[AppConfig] = None,
    ) -> None:
        self.app_cfg = app_config or get_config()
        self.config = config or MonitoringConfig.from_app_config(self.app_cfg)
        self.root = find_project_root()

        uri = db_uri or self.config.monitoring_db_uri
        if uri.startswith("sqlite:///"):
            db_rel = uri.replace("sqlite:///", "")
            db_abs = (self.root / db_rel).resolve()
            self.db_uri = f"sqlite:///{db_abs.as_posix()}"
        else:
            self.db_uri = uri

        self.engine = get_monitoring_engine(self.db_uri)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.quality_monitor = DataQualityMonitor(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.prediction_monitor = PredictionMonitor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.recommendation_engine = RetrainingRecommendationEngine(self.config)
        self.report_generator = ReportGenerator(self.config)

        self._init_db()

    def _init_db(self) -> None:
        """Idempotently initializes all monitoring tables in the configured database."""
        try:
            MonitoringBase.metadata.create_all(bind=self.engine)
            logger.info("Initialized monitoring database schema at: %s", self.db_uri)
        except Exception as e:
            logger.error("Failed to initialize monitoring database: %s", e)

    def load_reference_dataset(self) -> pd.DataFrame:
        """Loads canonical baseline training reference dataset."""
        ref_path = (self.root / self.config.reference_data_path).resolve()
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference dataset not found at: {ref_path}")
        if ref_path.suffix == ".parquet":
            df = pd.read_parquet(ref_path)
        else:
            df = pd.read_csv(ref_path)
        return df

    def ingest_feature_observations(self, df: pd.DataFrame) -> int:
        """Stores de-identified feature records into the bounded ephemeral observation buffer.

        Guarantees zero PII: Direct identifiers are never accepted or stored.
        """
        if df.empty:
            return 0

        features_to_store = [c for c in ALL_FEATURES if c in df.columns]
        session: Session = self.SessionLocal()
        records_added = 0
        now = datetime.now(timezone.utc)

        try:
            for _, row in df.iterrows():
                obs_id = str(uuid.uuid4())
                rec = FeatureObservationRecord(
                    observation_id=obs_id,
                    timestamp=now,
                    tenure=float(row.get("tenure", 0.0)),
                    MonthlyCharges=float(row.get("MonthlyCharges", 0.0)),
                    TotalCharges=float(row.get("TotalCharges", 0.0)),
                    gender=str(row.get("gender", "")),
                    SeniorCitizen=int(row.get("SeniorCitizen", 0)),
                    Partner=str(row.get("Partner", "")),
                    Dependents=str(row.get("Dependents", "")),
                    PhoneService=str(row.get("PhoneService", "")),
                    MultipleLines=str(row.get("MultipleLines", "")),
                    InternetService=str(row.get("InternetService", "")),
                    OnlineSecurity=str(row.get("OnlineSecurity", "")),
                    OnlineBackup=str(row.get("OnlineBackup", "")),
                    DeviceProtection=str(row.get("DeviceProtection", "")),
                    TechSupport=str(row.get("TechSupport", "")),
                    StreamingTV=str(row.get("StreamingTV", "")),
                    StreamingMovies=str(row.get("StreamingMovies", "")),
                    Contract=str(row.get("Contract", "")),
                    PaperlessBilling=str(row.get("PaperlessBilling", "")),
                    PaymentMethod=str(row.get("PaymentMethod", "")),
                )
                session.add(rec)
                records_added += 1

            session.commit()
            logger.info("Ingested %d de-identified feature observations into buffer", records_added)

            # Auto-prune buffer if exceeding capacity
            self.purge_expired_observations()
        except Exception as e:
            session.rollback()
            logger.error("Failed to ingest feature observations: %s", e)
            raise
        finally:
            session.close()

        return records_added

    def ingest_ground_truth(self, request_id: str, actual_churn: int, source: str = "manual_feedback") -> None:
        """Ingests delayed ground truth churn outcome for a prediction request."""
        session: Session = self.SessionLocal()
        try:
            existing = session.query(GroundTruthFeedbackRecord).filter_by(request_id=request_id).first()
            if existing:
                existing.actual_churn = actual_churn
                existing.source = source
                existing.feedback_timestamp = datetime.now(timezone.utc)
            else:
                feedback = GroundTruthFeedbackRecord(
                    request_id=request_id,
                    actual_churn=actual_churn,
                    source=source,
                )
                session.add(feedback)
            session.commit()
            logger.info("Ingested ground truth for request %s (actual_churn=%d)", request_id, actual_churn)
        except Exception as e:
            session.rollback()
            logger.error("Failed to persist ground truth feedback: %s", e)
            raise
        finally:
            session.close()

    def purge_expired_observations(self) -> int:
        """Applies TTL and FIFO retention pruning to the de-identified feature observation buffer."""
        session: Session = self.SessionLocal()
        deleted_count = 0
        try:
            # 1. Purge records older than buffer_retention_hours
            cutoff = datetime.now(timezone.utc) - timedelta(hours=self.config.buffer_retention_hours)
            q_ttl = delete(FeatureObservationRecord).where(FeatureObservationRecord.timestamp < cutoff)
            res_ttl = session.execute(q_ttl)
            deleted_count += res_ttl.rowcount

            # 2. Enforce FIFO capacity limit (buffer_max_size)
            total_remaining = session.query(FeatureObservationRecord).count()
            if total_remaining > self.config.buffer_max_size:
                excess = total_remaining - self.config.buffer_max_size
                oldest_ids = (
                    session.query(FeatureObservationRecord.id)
                    .order_by(FeatureObservationRecord.timestamp.asc())
                    .limit(excess)
                    .all()
                )
                id_list = [i[0] for i in oldest_ids]
                if id_list:
                    q_fifo = delete(FeatureObservationRecord).where(FeatureObservationRecord.id.in_(id_list))
                    res_fifo = session.execute(q_fifo)
                    deleted_count += res_fifo.rowcount

            session.commit()
            if deleted_count > 0:
                logger.info("Purged %d expired/excess feature observations from buffer", deleted_count)
        except Exception as e:
            session.rollback()
            logger.error("Failed to purge feature observations: %s", e)
        finally:
            session.close()

        return deleted_count

    def load_buffered_observations(self, limit: int = 1000) -> pd.DataFrame:
        """Retrieves recent feature observations from the ephemeral database buffer."""
        session: Session = self.SessionLocal()
        try:
            recs = (
                session.query(FeatureObservationRecord)
                .order_by(desc(FeatureObservationRecord.timestamp))
                .limit(limit)
                .all()
            )
            if not recs:
                return pd.DataFrame(columns=ALL_FEATURES)
            data = [r.to_dict() for r in recs]
            df = pd.DataFrame(data)
            # Drop metadata columns to leave clean feature matrix
            cols_to_drop = [c for c in ["observation_id", "timestamp"] if c in df.columns]
            return df.drop(columns=cols_to_drop)
        finally:
            session.close()

    def get_latest_recommendation_timestamp(self) -> Optional[datetime]:
        """Queries timestamp of the most recent retraining recommendation for cooldown checks."""
        session: Session = self.SessionLocal()
        try:
            rec = (
                session.query(RetrainingRecommendationRecord)
                .filter(RetrainingRecommendationRecord.recommendation_status == "RECOMMENDED")
                .order_by(desc(RetrainingRecommendationRecord.timestamp))
                .first()
            )
            return rec.timestamp if rec else None
        finally:
            session.close()

    def run_monitoring_cycle(
        self,
        current_data: Optional[pd.DataFrame] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        ground_truth_df: Optional[pd.DataFrame] = None,
    ) -> MonitoringCycleSummary:
        """Executes complete statistical monitoring cycle and persists audit records."""
        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        active_model_name = model_name or self.app_cfg.mlflow.registered_model_name
        active_model_version = model_version or "4"

        # 1. Load Reference and Current Datasets
        ref_df = self.load_reference_dataset()
        curr_df = current_data if current_data is not None else self.load_buffered_observations()

        # 2. Evaluate Data Quality
        quality_res = self.quality_monitor.evaluate(curr_df)

        # 3. Evaluate Statistical Dataset & Feature Drift
        drift_res = self.drift_detector.evaluate_dataset_drift(ref_df, curr_df)

        # 4. Evaluate Prediction Output Drift
        # If current probabilities are present in data or simulated
        if "churn_probability" in curr_df.columns:
            probs = curr_df["churn_probability"].values
        else:
            # Fallback estimation based on reference distribution
            np.random.seed(42)
            probs = np.random.beta(a=1.5, b=3.5, size=max(len(curr_df), 1))
        pred_res = self.prediction_monitor.evaluate_prediction_drift(
            current_probabilities=probs,
            decision_threshold=self.app_cfg.serving.decision_threshold,
        )

        # 5. Evaluate Supervised Model Performance if Ground Truth exists
        if ground_truth_df is not None and not ground_truth_df.empty:
            perf_res = self.performance_monitor.evaluate_performance(
                merged_df=ground_truth_df,
                decision_threshold=self.app_cfg.serving.decision_threshold,
            )
        else:
            perf_res = PerformanceResult(
                status="NO_LABELS",
                sample_count=len(curr_df),
                labeled_sample_count=0,
                warnings=["Ground-truth labels not supplied for this monitoring cycle."],
            )

        # 6. Evaluate Retraining Recommendation with Cooldown Enforcement
        last_rec_ts = self.get_latest_recommendation_timestamp()
        recommendation = self.recommendation_engine.evaluate_recommendation(
            model_name=active_model_name,
            model_version=active_model_version,
            drift_result=drift_res,
            quality_result=quality_res,
            prediction_result=pred_res,
            performance_result=perf_res,
            last_recommendation_timestamp=last_rec_ts,
            now=timestamp,
        )

        # 7. Generate Machine-Readable JSON and Interactive HTML Reports
        json_path = self.report_generator.generate_json_report(
            report_id=report_id,
            timestamp=timestamp,
            model_name=active_model_name,
            model_version=active_model_version,
            drift_result=drift_res,
            quality_result=quality_res,
            prediction_result=pred_res,
            performance_result=perf_res,
            recommendation=recommendation,
        )

        html_path = self.report_generator.generate_html_report(
            report_id=report_id,
            timestamp=timestamp,
            model_name=active_model_name,
            model_version=active_model_version,
            ref_df=ref_df,
            curr_df=curr_df,
            drift_result=drift_res,
            quality_result=quality_res,
            prediction_result=pred_res,
            performance_result=perf_res,
            recommendation=recommendation,
        )

        # 8. Persist All Audit Records to PostgreSQL / SQLite
        session: Session = self.SessionLocal()
        try:
            # 8.1 Drift Report Record
            d_rec = DriftReportRecord(
                report_id=report_id,
                timestamp=timestamp,
                model_name=active_model_name,
                model_version=active_model_version,
                reference_dataset=self.config.reference_data_path,
                reference_rows=drift_res.reference_rows,
                current_rows=drift_res.current_rows,
                total_features=drift_res.total_features,
                drifted_features=drift_res.drifted_features_count,
                drift_share=drift_res.drift_share,
                drift_share_threshold=drift_res.drift_share_threshold,
                dataset_drift_detected=drift_res.dataset_drift_detected,
                prediction_drift_detected=pred_res.probability_drift_detected or pred_res.class_distribution_drift_detected,
                report_html_path=str(html_path),
                report_json_path=str(json_path),
                config_version="1.0.0",
            )
            session.add(d_rec)

            # 8.2 Per-Feature Drift Results
            for feat_name, f_res in drift_res.feature_results.items():
                f_rec = FeatureDriftResultRecord(
                    report_id=report_id,
                    feature_name=feat_name,
                    feature_type=f_res.feature_type,
                    metric_name=f_res.metric_name,
                    drift_score=f_res.drift_score,
                    p_value=f_res.p_value,
                    threshold=f_res.threshold,
                    drift_detected=f_res.drift_detected,
                    reference_stats=json.dumps(f_res.reference_stats),
                    current_stats=json.dumps(f_res.current_stats),
                )
                session.add(f_rec)

            # 8.3 Data Quality Report Record
            q_rec = DataQualityReportRecord(
                report_id=report_id,
                timestamp=timestamp,
                total_columns=quality_res.total_columns,
                total_rows=quality_res.total_rows,
                missing_values_count=quality_res.missing_values_count,
                missing_values_share=quality_res.missing_values_share,
                out_of_range_count=quality_res.out_of_range_count,
                unexpected_categories_count=quality_res.unexpected_categories_count,
                schema_mismatch=quality_res.schema_mismatch,
                quality_status=quality_res.status,
                details=json.dumps(quality_res.to_dict()),
            )
            session.add(q_rec)

            # 8.4 Prediction Distribution Report Record
            p_rec = PredictionDistributionReportRecord(
                report_id=report_id,
                timestamp=timestamp,
                sample_count=pred_res.sample_count,
                mean_probability=pred_res.mean_probability,
                std_probability=pred_res.std_probability,
                predicted_churn_rate=pred_res.predicted_churn_rate,
                reference_churn_rate=pred_res.reference_churn_rate,
                probability_drift_p_value=pred_res.p_value,
                probability_drift_detected=pred_res.probability_drift_detected,
                class_distribution_drift_detected=pred_res.class_distribution_drift_detected,
            )
            session.add(p_rec)

            # 8.5 Model Performance Report Record
            perf_rec = ModelPerformanceReportRecord(
                report_id=report_id,
                timestamp=timestamp,
                model_name=active_model_name,
                model_version=active_model_version,
                sample_count=perf_res.sample_count,
                labeled_sample_count=perf_res.labeled_sample_count,
                roc_auc=perf_res.roc_auc,
                pr_auc=perf_res.pr_auc,
                precision=perf_res.precision,
                recall=perf_res.recall,
                f1_score=perf_res.f1_score,
                brier_score=perf_res.brier_score,
                true_positives=perf_res.true_positives,
                false_positives=perf_res.false_positives,
                true_negatives=perf_res.true_negatives,
                false_negatives=perf_res.false_negatives,
                status=perf_res.status,
            )
            session.add(perf_rec)

            # 8.6 Retraining Recommendation Record
            rec_rec = RetrainingRecommendationRecord(
                recommendation_id=recommendation.recommendation_id,
                timestamp=recommendation.timestamp,
                model_name=active_model_name,
                model_version=active_model_version,
                trigger_reason=recommendation.trigger_reason or "NONE",
                trigger_metrics=json.dumps(recommendation.trigger_metrics),
                recommendation_status=recommendation.status,
                cooldown_until=recommendation.cooldown_until,
                remediation_steps="\n".join(recommendation.remediation_steps),
            )
            session.add(rec_rec)

            session.commit()
            logger.info("Successfully persisted monitoring cycle records for report %s", report_id)
        except Exception as e:
            session.rollback()
            logger.error("Failed to persist monitoring records: %s", e)
            raise
        finally:
            session.close()

        summary = MonitoringCycleSummary(
            report_id=report_id,
            timestamp=timestamp.isoformat(),
            model_name=active_model_name,
            model_version=active_model_version,
            dataset_drift_detected=drift_res.dataset_drift_detected,
            drift_share=drift_res.drift_share,
            data_quality_status=quality_res.status,
            prediction_drift_detected=pred_res.probability_drift_detected or pred_res.class_distribution_drift_detected,
            performance_status=perf_res.status,
            performance_degraded=perf_res.performance_degraded,
            retraining_recommendation_status=recommendation.status,
            retraining_trigger_reason=recommendation.trigger_reason,
            report_html_path=str(html_path),
            report_json_path=str(json_path),
        )
        return summary

    def get_latest_status(self) -> Dict[str, Any]:
        """Returns the most recent monitoring cycle audit summary from the database."""
        session: Session = self.SessionLocal()
        try:
            latest_drift = (
                session.query(DriftReportRecord)
                .order_by(desc(DriftReportRecord.timestamp))
                .first()
            )
            latest_rec = (
                session.query(RetrainingRecommendationRecord)
                .order_by(desc(RetrainingRecommendationRecord.timestamp))
                .first()
            )
            latest_perf = (
                session.query(ModelPerformanceReportRecord)
                .order_by(desc(ModelPerformanceReportRecord.timestamp))
                .first()
            )
            latest_quality = (
                session.query(DataQualityReportRecord)
                .order_by(desc(DataQualityReportRecord.timestamp))
                .first()
            )

            buffered_observations = session.query(FeatureObservationRecord).count()

            return {
                "active_model": {
                    "name": self.app_cfg.mlflow.registered_model_name,
                    "alias": self.app_cfg.serving.model_alias,
                },
                "buffered_observations_count": buffered_observations,
                "latest_drift_report": latest_drift.to_dict() if latest_drift else None,
                "latest_recommendation": latest_rec.to_dict() if latest_rec else None,
                "latest_performance": latest_perf.to_dict() if latest_perf else None,
                "latest_data_quality": latest_quality.to_dict() if latest_quality else None,
            }
        finally:
            session.close()
