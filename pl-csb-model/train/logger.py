from transformers.utils import logging
from typing import Optional

def set_up_logger(name: str = "transformers", level: Optional[int] = None):
    if level is not None:
        logging.set_verbosity(level)
    else:
        logging.set_verbosity_info()
    logger = logging.get_logger(name)

    return logger
