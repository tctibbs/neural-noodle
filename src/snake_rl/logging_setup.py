"""Loguru configuration with a run id bound to every log line."""

import sys
import uuid
from pathlib import Path

from loguru import logger


def setup_logging(
    run_id: str | None = None,
    log_dir: Path | None = None,
    level: str = "INFO",
) -> str:
    """Configure loguru for a run and return the run id.

    Every log line carries the run id so interleaved runs stay
    attributable. If a log directory is given, a per-run file sink is
    added alongside stderr.

    Args:
        run_id: Identifier for this run. Generated when omitted.
        log_dir: Directory for the per-run log file. No file sink when
            omitted.
        level: Minimum log level for both sinks.

    Returns:
        The run id bound to the logger.
    """
    rid = run_id or uuid.uuid4().hex[:10]
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "{extra[run_id]} | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | {message}"
    )
    logger.configure(extra={"run_id": rid})
    logger.add(sys.stderr, format=fmt, level=level)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / f"{rid}.log",
            format=fmt,
            level=level,
            enqueue=True,
        )
    return rid
