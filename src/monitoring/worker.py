"""Executable CLI worker for scheduled and on-demand model monitoring cycles."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
from typing import Optional
import pandas as pd

from src.common.logger import get_logger
from src.monitoring.monitoring_config import MonitoringConfig
from src.monitoring.monitoring_service import MonitoringService

logger = get_logger("src.monitoring.worker")

# Global flag for graceful daemon shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    logger.info("Received signal %d; shutting down monitoring worker gracefully...", signum)
    _shutdown_requested = True


def print_status(service: MonitoringService) -> None:
    """Displays formatted current monitoring and drift status."""
    status = service.get_latest_status()
    print("\n" + "=" * 70)
    print(" Enterprise MLOps Platform — Monitoring Status")
    print("=" * 70)
    print(f"Active Model: {status['active_model']['name']} (alias: {status['active_model']['alias']})")
    print(f"Buffered Feature Observations: {status['buffered_observations_count']}")

    d_rep = status.get("latest_drift_report")
    if d_rep:
        print("\n[Latest Drift Evaluation]")
        print(f"  Report ID: {d_rep['report_id']}")
        print(f"  Timestamp: {d_rep['timestamp']}")
        print(f"  Drift Detected: {d_rep['dataset_drift_detected']}")
        print(f"  Drift Share: {d_rep['drift_share'] * 100:.1f}% (Threshold: {d_rep['drift_share_threshold'] * 100:.1f}%)")
        print(f"  Drifted Features: {d_rep['drifted_features']} / {d_rep['total_features']}")
        print(f"  HTML Report: {d_rep['report_html_path']}")
        print(f"  JSON Report: {d_rep['report_json_path']}")
    else:
        print("\n[Latest Drift Evaluation]: No reports recorded yet.")

    q_rep = status.get("latest_data_quality")
    if q_rep:
        print("\n[Data Quality]")
        print(f"  Status: {q_rep['quality_status']}")
        print(f"  Missing Value Share: {q_rep['missing_values_share'] * 100:.2f}%")
        print(f"  Out of Range Count: {q_rep['out_of_range_count']}")
        print(f"  Unexpected Categories: {q_rep['unexpected_categories_count']}")

    rec = status.get("latest_recommendation")
    if rec:
        print("\n[Retraining Recommendation]")
        print(f"  Status: {rec['recommendation_status']}")
        print(f"  Trigger Reason: {rec['trigger_reason']}")
        print(f"  Cooldown Until: {rec['cooldown_until'] or 'Inactive'}")

    perf = status.get("latest_performance")
    if perf:
        print("\n[Supervised Performance]")
        print(f"  Status: {perf['status']}")
        if perf["status"] == "EVALUATED":
            print(f"  ROC-AUC: {perf['roc_auc']:.4f}")
            print(f"  PR-AUC: {perf['pr_auc']:.4f}")
            print(f"  Recall: {perf['recall']:.4f}")
            print(f"  F1: {perf['f1_score']:.4f}")
    print("=" * 70 + "\n")


def run_once(service: MonitoringService, data_path: Optional[str] = None) -> int:
    """Executes a single monitoring cycle."""
    logger.info("Executing on-demand monitoring cycle...")
    curr_df = None
    if data_path:
        p = Path(data_path)
        if not p.exists():
            logger.error("Specified data file does not exist: %s", data_path)
            return 1
        curr_df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    summary = service.run_monitoring_cycle(current_data=curr_df)
    print("\n" + "-" * 70)
    print(" Monitoring Cycle Complete")
    print("-" * 70)
    print(f"Report ID: {summary.report_id}")
    print(f"Dataset Drift: {'DETECTED' if summary.dataset_drift_detected else 'NOMINAL'} (Share: {summary.drift_share*100:.1f}%)")
    print(f"Data Quality: {summary.data_quality_status}")
    print(f"Prediction Drift: {'DETECTED' if summary.prediction_drift_detected else 'NOMINAL'}")
    print(f"Retraining Guidance: {summary.retraining_recommendation_status} (Trigger: {summary.retraining_trigger_reason or 'None'})")
    print(f"HTML Report: {summary.report_html_path}")
    print(f"JSON Report: {summary.report_json_path}")
    print("-" * 70 + "\n")
    return 0


def run_daemon(service: MonitoringService, interval_seconds: int = 3600) -> None:
    """Runs periodic monitoring jobs at the specified interval."""
    logger.info("Starting monitoring worker daemon (interval=%d seconds)...", interval_seconds)
    while not _shutdown_requested:
        try:
            logger.info("Running scheduled monitoring cycle at %s", datetime.now(timezone.utc).isoformat())
            service.run_monitoring_cycle()
        except Exception as e:
            logger.error("Error during scheduled monitoring cycle: %s", e)

        # Sleep in small increments to respond promptly to shutdown signals
        for _ in range(interval_seconds):
            if _shutdown_requested:
                break
            time.sleep(1)

    logger.info("Monitoring worker daemon terminated cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Enterprise MLOps Platform — Monitoring Worker CLI")
    parser.add_argument("--run-once", action="store_true", help="Execute a single monitoring cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in background daemon mode")
    parser.add_argument("--interval-seconds", type=int, default=3600, help="Interval between daemon cycles in seconds")
    parser.add_argument("--status", action="store_true", help="Display latest monitoring status and exit")
    parser.add_argument("--data-file", type=str, default=None, help="Optional data file to evaluate for drift")
    parser.add_argument("--ingest-observations", type=str, default=None, help="Path to CSV/Parquet file to ingest into feature buffer")
    parser.add_argument("--ingest-ground-truth", type=str, default=None, help="Path to CSV file with request_id and actual_churn")
    parser.add_argument("--purge", action="store_true", help="Purge expired observations from the ephemeral buffer")
    parser.add_argument("--db-uri", type=str, default=None, help="Database URI override")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    service = MonitoringService(db_uri=args.db_uri)

    if args.purge:
        deleted = service.purge_expired_observations()
        print(f"Purged {deleted} expired observations.")
        return 0

    if args.ingest_observations:
        p = Path(args.ingest_observations)
        if not p.exists():
            print(f"File not found: {p}")
            return 1
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        count = service.ingest_feature_observations(df)
        print(f"Ingested {count} de-identified feature observations into buffer.")
        return 0

    if args.ingest_ground_truth:
        p = Path(args.ingest_ground_truth)
        if not p.exists():
            print(f"File not found: {p}")
            return 1
        df = pd.read_csv(p)
        if "request_id" not in df.columns or "actual_churn" not in df.columns:
            print("CSV must contain 'request_id' and 'actual_churn' columns.")
            return 1
        for _, row in df.iterrows():
            service.ingest_ground_truth(str(row["request_id"]), int(row["actual_churn"]))
        print(f"Ingested {len(df)} ground-truth feedback records.")
        return 0

    if args.status:
        print_status(service)
        return 0

    if args.daemon:
        run_daemon(service, interval_seconds=args.interval_seconds)
        return 0

    # Default action if no specific mode selected is --run-once
    return run_once(service, data_path=args.data_file)


if __name__ == "__main__":
    sys.exit(main())
