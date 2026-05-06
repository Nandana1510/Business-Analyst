"""Pipeline and LLM agent visibility on stderr (works under uvicorn)."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager

_LOGGER_NAME = "ba.pipeline"
_configured = False


def configure_pipeline_logging() -> None:
    """
    Attach a single stderr handler for ``ba.pipeline`` (idempotent).

    Set ``PIPELINE_LOG_LEVEL`` to DEBUG, INFO, WARNING, or ERROR (default: INFO).
    """
    global _configured
    if _configured:
        return
    _configured = True
    raw = (os.environ.get("PIPELINE_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, raw, logging.INFO)
    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(level)
    log.handlers.clear()
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter("%(asctime)s [ba.pipeline] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    log.addHandler(h)
    log.propagate = False


def pipeline_log() -> logging.Logger:
    """Logger used for session-level milestones and :func:`agent_log` spans."""
    configure_pipeline_logging()
    return logging.getLogger(_LOGGER_NAME)


@contextmanager
def agent_log(name: str):
    """Log ``[START]`` / ``[END]`` around one LLM-backed agent call."""
    log = pipeline_log()
    log.info("[START] %s", name)
    try:
        yield
    finally:
        log.info("[END] %s", name)
