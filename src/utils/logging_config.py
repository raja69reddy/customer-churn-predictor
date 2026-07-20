"""Shared logging setup — every src/ script logs to both the console and logs/<name>.log."""

import logging
import os

LOGS_DIR = "logs"


def setup_logging(name: str) -> logging.Logger:
    """Returns a logger that writes to stderr and to logs/<name>.log."""
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers on re-import

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"{name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
