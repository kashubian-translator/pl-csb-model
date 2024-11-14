import logging
from typing import Optional


map_level = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG
}

DEFAULT_LOGGER_LEVEL = logging.DEBUG


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
