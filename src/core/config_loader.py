from pathlib import Path

import yaml

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IConfigLoader, IConfigValidator

logger = get_logger(__name__)


class ConfigLoader(IConfigLoader):
    """Loads and validates YAML configuration files."""

    def __init__(self, path: str, validator: IConfigValidator) -> None:
        """Initialize loader with file path and validator."""
        self._path = Path(path)
        self._validator = validator

    def read(self) -> dict:
        """Read YAML file, validate structure, and return data."""
        logger.debug(f"Loading config file: {self._path}")
        with self._path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._validator.validate_root(data)
        return data
