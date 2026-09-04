"""Local logging configuration for SIGANALYZER."""

import logging
import os
from pathlib import Path
import sys


def setup_logger(
    name: str = "siganalyzer",
    log_dir: Path | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a local rotating logger.

    Logs are written strictly to local storage. Zero network telemetry.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # Local file handler
    candidates = [
        log_dir or Path.home() / ".siganalyzer" / "logs",
        Path.cwd() / ".siganalyzer" / "logs",
    ]

    for target_dir in candidates:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            log_file = target_dir / "siganalyzer.log"
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=5 * 1024 * 1024,  # 5 MB
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            break
        except Exception:
            continue

    return logger


logger = setup_logger()
