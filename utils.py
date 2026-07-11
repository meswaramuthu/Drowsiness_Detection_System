"""
=============================================================================
  utils.py — Utility & Helper Functions
=============================================================================
  Shared helpers used across the project:
    - Logging setup (file + console)
    - Timestamp formatting
    - Directory initialization
    - FPS calculation
=============================================================================
"""

import os
import logging
import time
from datetime import datetime

import config


# ──────────────────────────────────────────────────────────────────────────────
# Directory Initialization
# ──────────────────────────────────────────────────────────────────────────────

def ensure_directories() -> None:
    """
    Create all required project directories if they don't already exist.
    Called once at application startup.
    """
    dirs = [config.OUTPUT_DIR, config.LOG_DIR, config.ASSETS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "drowsiness") -> logging.Logger:
    """
    Configure and return a logger that writes to both console and a
    timestamped log file in the logs/ directory.

    Args:
        name: Logger name (used to retrieve the same logger across modules).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # ── Log format ──
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # ── File handler (one log file per session) ──
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{config.LOG_FILE_PREFIX}_{timestamp}.log"
    log_path = os.path.join(config.LOG_DIR, log_filename)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Log file created: {log_path}")
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Timestamp Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    """Return a human-readable timestamp string for display overlays."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_file_timestamp() -> str:
    """Return a filename-safe timestamp string (no colons or spaces)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ──────────────────────────────────────────────────────────────────────────────
# FPS Calculator
# ──────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """
    A lightweight FPS (frames-per-second) tracker.

    Usage:
        fps = FPSCounter()
        while True:
            fps.tick()
            current_fps = fps.get()
    """

    def __init__(self, smoothing: int = 30):
        """
        Args:
            smoothing: Number of recent frames to average over.
                       Higher = smoother but slower to react.
        """
        self._smoothing = smoothing
        self._timestamps: list[float] = []

    def tick(self) -> None:
        """Record a frame timestamp. Call once per frame."""
        now = time.perf_counter()
        self._timestamps.append(now)

        # Keep only the most recent N timestamps
        if len(self._timestamps) > self._smoothing:
            self._timestamps = self._timestamps[-self._smoothing:]

    def get(self) -> float:
        """
        Calculate and return the current FPS.

        Returns:
            Frames per second (float). Returns 0.0 if fewer than 2 frames
            have been recorded.
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed
