from typing import Any

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IConfigValidator

logger = get_logger(__name__)


class ConfigValidator(IConfigValidator):
    """Validate structure and content of YAML configuration."""

    def validate_root(self, data: Any) -> None:
        """Validate the YAML top-level structure."""
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping.")
        if not ("experiments" in data or "sweep" in data):
            raise ValueError("YAML must contain either 'experiments' or 'sweep'.")
        logger.debug("Root configuration validated successfully.")

    def validate_problem(self, problem: dict) -> None:
        """Validate the problem section."""
        if "file_path" not in problem or "optimal_results_path" not in problem:
            raise ValueError("Problem section must contain 'file_path' and 'optimal_results_path'.")
        logger.debug("Problem configuration validated successfully.")

    def validate_algorithm(self, algorithm: dict[str, Any], allow_lists: bool) -> None:
        """Validate algorithm configuration based on algorithm type."""
        algo_name = algorithm.get("name")

        if algo_name == "ga":
            self._validate_algorithm_ga(algorithm, allow_lists)
        elif algo_name == "acs":
            self._validate_algorithm_acs(algorithm, allow_lists)
        else:
            raise ValueError(f"Unknown algorithm type: {algo_name}")

        logger.debug("Algorithm configuration validated successfully.")

    def _validate_algorithm_ga(self, algorithm: dict[str, Any], allow_lists: bool) -> None:
        """Validate GA configuration fields."""
        required = [
            "population_size",
            "crossover_rate",
            "mutation_rate",
            "max_time",
            "selection_config",
            "crossover_config",
            "mutation_config",
            "succession_config",
        ]

        operator_keys = [
            "selection_config",
            "crossover_config",
            "mutation_config",
            "succession_config",
        ]

        for r in required:
            if r not in algorithm:
                if r in operator_keys:
                    raise ValueError(f"Algorithm configuration must include {r} section")
                raise ValueError(f"Missing required algorithm field: {r}")

        if not allow_lists:
            for key, val in algorithm.items():
                if isinstance(val, list):
                    raise ValueError(f"Unexpected list in algorithm config for field: {key}")

    def _validate_algorithm_acs(self, algorithm: dict[str, Any], allow_lists: bool) -> None:
        """Validate ACS configuration fields."""
        required = [
            "num_ants",
            "alpha",
            "beta",
            "rho",
            "phi",
            "q0",
            "max_time",
        ]

        for r in required:
            if r not in algorithm:
                raise ValueError(f"Missing required ACS field: {r}")

        forbidden = [
            "population_size",
            "crossover_rate",
            "mutation_rate",
            "selection_config",
            "crossover_config",
            "mutation_config",
            "succession_config",
        ]

        for f in forbidden:
            if f in algorithm:
                raise ValueError("Unexpected GA field in ACS")

        if not allow_lists:
            for key, val in algorithm.items():
                if isinstance(val, list):
                    raise ValueError(f"Unexpected list in algorithm config for field: {key}")
