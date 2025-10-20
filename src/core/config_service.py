from typing import Optional

from src.core.logger import get_logger
from src.interfaces.core_interfaces import (
    IConfigExpander,
    IConfigLoader,
    IConfigService,
    IConfigValidator,
)

logger = get_logger(__name__)


class ConfigService(IConfigService):
    """Singleton managing configuration loading and expansion."""

    _instance: Optional["ConfigService"] = None

    def __new__(cls, *args, **kwargs):
        """Ensure a single instance of the service exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, loader: IConfigLoader, validator: IConfigValidator, expander: IConfigExpander
    ):
        """Initialize service with loader, validator, and expander."""
        self._loader = loader
        self._validator = validator
        self._expander = expander
        self._configs = None

    def load_all(self):
        """Load, validate, expand, and return all experiment configurations."""
        raw = self._loader.read()
        self._configs = self._expander.expand(raw)
        logger.info(f"Loaded {len(self._configs)} experiment configurations.")
        return self._configs
