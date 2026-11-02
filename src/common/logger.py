"""Structured Logging Module for Enterprise MLOps Platform."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "mlops_platform", log_level: Optional[str] = None) -> logging.Logger:
    """Configures and returns a standardized, thread-safe logger.

    Args:
        name: Hierarchical module name (e.g. `src.data.loader`).
        log_level: Optional log level override (e.g. "INFO", "DEBUG").

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_name = (log_level or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
