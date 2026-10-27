"""Project-specific rotating debug log for batch and GUI error inspection."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger():
    return logging.getLogger('agar_colony')


def setup_file_logging(path):
    """Attach one rotating file handler and avoid duplicating logs on repeated runs."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger()
    logger.setLevel(logging.INFO)
    resolved = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and handler.baseFilename == resolved:
            return logger
    handler = RotatingFileHandler(resolved, maxBytes=2_000_000, backupCount=2,
                                  encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    return logger
