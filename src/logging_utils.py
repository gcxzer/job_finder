from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.configs import CONFIG


def setup_logger(log_file: str | Path | None = None) -> logging.Logger:
    log_path = Path(log_file) if log_file else _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("job_finder")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _default_log_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    return CONFIG.workspace.logs_dir / f"job_finder_{timestamp}.log"
