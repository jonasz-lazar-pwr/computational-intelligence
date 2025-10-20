import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=2_000_000,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))

console_handler = RichHandler(
    markup=True,
    rich_tracebacks=True,
    tracebacks_show_locals=False,
    show_time=True,
    show_level=True,
    show_path=False,
)
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler],
    force=True,
)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)


def disable_file_logging() -> None:
    """Disable all file-based logging handlers."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
    logging.getLogger().disabled = False
    logging.debug("File logging disabled.")


def enable_file_logging() -> None:
    """Re-enable file logging (after it was disabled)."""
    root = logging.getLogger()
    has_file_handler = any(isinstance(h, logging.FileHandler) for h in root.handlers)
    if not has_file_handler:
        root.addHandler(file_handler)
    logging.debug("File logging re-enabled.")
