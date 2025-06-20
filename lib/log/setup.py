import logging
import os
import sys
import datetime as dt


def setup_logging(log_level=logging.INFO, log_file_path: str = None):
    """
    Configures the root logger with timestamp and module name

    Args:
        log_level: Logging level (default: logging.INFO)
        log_folder: Optional folder path to write logs (default: None)
    """

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s]-[%(name)s]-[%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file_path:
        now = dt.datetime.now()
        file_name_time_part = now.strftime(f"%Y-%m-%d_%H-%M-%S")
        folder_path = os.path.dirname(log_file_path)
        base_path, ext = os.path.splitext(log_file_path)
        file_path = f"{base_path}-{file_name_time_part}.{ext}"

        # Create logs directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)