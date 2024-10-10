import logging
from typing import Optional


map_level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

DEFAULT_LOGGER_LEVEL = logging.INFO


def set_up_logger(name: str, logger_level: Optional[str] = None) -> logging.Logger:
    level = map_level.get(logger_level, DEFAULT_LOGGER_LEVEL) if logger_level else DEFAULT_LOGGER_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s [%(process)s] [%(levelname)s] - %(message)s")

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger
