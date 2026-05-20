"""
core/logging_setup.py
Centralised loguru configuration. Call setup_logging() once at startup.
All modules use: from loguru import logger
"""
import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "00:00",
    retention: str = "30 days",
    diagnose: bool = False,
) -> None:
    """
    Configure loguru for the trading bot.

    Outputs:
      - Colourised stdout (human-readable)
      - Rotating daily JSON log file (machine-parseable)
      - Separate error log file
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # ---- Stdout: human-readable with colour ----
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        diagnose=diagnose,
    )

    # ---- Rotating daily log file (JSON) ----
    logger.add(
        log_path / "trading_bot_{time:YYYY-MM-DD}.log",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        compression="gz",
        serialize=False,
        diagnose=diagnose,
    )

    # ---- Error-only log ----
    logger.add(
        log_path / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        diagnose=True,
    )

    logger.info("Logging initialised — level={} dir={}", level, log_path.resolve())
