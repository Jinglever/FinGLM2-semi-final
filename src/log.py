"""
This module provides custom logging handlers that write log messages without newlines.
"""

import logging
import threading
from typing import Optional

class NoNewlineStreamHandler(logging.StreamHandler):
    """
    A custom stream handler that writes log messages without adding newlines.
    """
    def emit(self, record):
        msg = self.format(record)
        stream = self.stream
        stream.write(msg)
        self.flush()

class NoNewlineFileHandler(logging.FileHandler):
    """
    A custom file handler that writes log messages without adding newlines.
    """
    def emit(self, record):
        msg = self.format(record)
        with open(self.baseFilename, self.mode, encoding=self.encoding) as file:
            file.write(msg)
            file.flush()

def get_logger(logger_name: Optional[str] = None):
    """
    Get a custom logger.

    :param logger_name: Name of the logger to update (default is 'logger_<thread_id>').
    :return: Configured logger instance.
    """
    if logger_name is None:
        thread_id = threading.get_ident()
        logger_name = f"logger_{thread_id}"
    return logging.getLogger(logger_name)

def setup_logger(log_file: str,
                 log_level: int = logging.INFO,
                 logger_name: Optional[str] = None):
    """
    Set the log file for the specified logger.

    :param log_file: file path for logging output.
    :param log_level: log level (default is 'logging.INFO')
    :param logger_name: Name of the logger to update (default is 'logger_<thread_id>').
    """
    if logger_name is None:
        thread_id = threading.get_ident()
        logger_name = f"logger_{thread_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Remove existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # Add new file handler
    formatter = logging.Formatter('%(message)s')
    new_file_handler = NoNewlineFileHandler(log_file, mode='a', encoding='utf-8')
    new_file_handler.setFormatter(formatter)
    logger.addHandler(new_file_handler)
