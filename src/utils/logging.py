"""Logging initialization and rich console formatting."""

import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logger(
    name: str = "learning_games",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up structured logger with Rich handler and optional file output.

    Parameters
    ----------
    name : str
        Logger identifier.
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : Optional[str]
        Filepath for logging output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(numeric_level)
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
