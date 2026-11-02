"""
Central logging configuration for INOOR.

Usage:
    from utils.logging_config import get_logger
    log = get_logger(__name__)
    log.warning("Failed to read header for %s: %s", path, e)

setup_logging() is called once from main.py. It installs:
  - a console handler (so developers still see messages in a terminal), and
  - a rotating file handler at Output/logs/inoor.log (5 files x 2 MB), so a
    packaged .exe build -- where print() output is invisible -- remains
    debuggable at the observatory.
"""
import logging
import logging.handlers
import os
import sys

_CONFIGURED = False


def _log_dir():
    """Output/logs next to the exe when frozen, else under the project root."""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'Output', 'logs')


def setup_logging(level=logging.INFO):
    """Idempotent root-logger configuration (console + rotating file)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        '%(asctime)s %(levelname)-7s [%(name)s] %(message)s', '%H:%M:%S')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    try:
        log_dir = _log_dir()
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'inoor.log'),
            maxBytes=2_000_000, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s [%(name)s] %(message)s'))
        root.addHandler(file_handler)
    except Exception as e:
        # A read-only install dir must not prevent the app from starting.
        console.setLevel(level)
        root.warning("File logging unavailable: %s", e)


def get_logger(name):
    return logging.getLogger(name)
