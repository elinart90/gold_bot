"""
utils/logger.py — Centralized logger. Import this in every module.
Usage:  from utils.logger import log
"""
import sys
from loguru import logger
from config.settings import LOGS_DIR

logger.remove()  # remove default handler

# Console — clean, coloured output
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan> — {message}",
    level="INFO",
)

# File — full detail, rotates daily
logger.add(
    LOGS_DIR / "bot_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{line} — {message}",
    level="DEBUG",
)

log = logger
