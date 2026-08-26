# import logging
# import os

# def setup_logging():

#     os.makedirs("logs", exist_ok=True)

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)

#     formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

#     file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
#     file_handler.setFormatter(formatter)

#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

import logging
import os
from datetime import datetime


def setup_logging():

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # ---------------------------------------------------------
    # Create timestamp for this application run
    # ---------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = os.path.join(
        "logs",
        f"{timestamp}.log"
    )

    # ---------------------------------------------------------
    # Root logger
    # ---------------------------------------------------------

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    # ---------------------------------------------------------
    # Prevent duplicate handlers
    #
    # This is important because FastAPI/Uvicorn can
    # initialize/reload application code more than once.
    # ---------------------------------------------------------

    if logger.handlers:
        logger.handlers.clear()

    # ---------------------------------------------------------
    # Log format
    # ---------------------------------------------------------

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    # ---------------------------------------------------------
    # File handler
    # ---------------------------------------------------------

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # Console handler
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # Add handlers
    # ---------------------------------------------------------

    logger.addHandler(
        file_handler
    )

    logger.addHandler(
        console_handler
    )

    # ---------------------------------------------------------
    # Startup log
    # ---------------------------------------------------------

    logger.info(
        "=================================================="
    )

    logger.info(
        "Application logging started"
    )

    logger.info(
        "Log file: %s",
        log_file
    )

    logger.info(
        "=================================================="
    )

    return log_file