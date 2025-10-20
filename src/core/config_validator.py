from typing import Any

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IConfigValidator

logger = get_logger(__name__)


class ConfigValidator(IConfigValidator):
    """Validates structure and content of YAML configuration."""

    def validate_root(self, data: Any) -> None:
        """Validate top-level YAML structure."""
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping.")
        if not ("experiments" in data or "sweep" in data):
            raise ValueError("YAML must contain either 'experiments' or 'sweep'.")
        logger.debug("Root configuration validated successfully.")

    def validate_problem(self, problem: dict) -> None:
        """Validate problem section fields."""
        if "file_path" not in problem or "optimal_results_path" not in problem:
            raise ValueError("Problem section must contain 'file_path' and 'optimal_results_path'.")
        logger.debug("Problem configuration validated successfully.")

    def validate_algorithm(self, algorithm: dict[str, Any], allow_lists: bool) -> None:
        """Validate algorithm parameters and operator sections."""
        required_fields = ["name", "population_size", "crossover_rate", "mutation_rate", "max_time"]
        for field in required_fields:
            if field not in algorithm:
                raise ValueError(f"Missing required algorithm field: {field}")
        for op in ["selection_config", "crossover_config", "mutation_config", "succession_config"]:
            if op not in algorithm:
                raise ValueError(f"Algorithm configuration must include {op} section.")
        if not allow_lists:
            for key, value in algorithm.items():
                if isinstance(value, list):
                    raise ValueError(f"Unexpected list in algorithm config for field: {key}")
        logger.debug("Algorithm configuration validated successfully.")
