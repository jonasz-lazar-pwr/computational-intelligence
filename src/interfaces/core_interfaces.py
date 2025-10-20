from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from src.core.models import ExperimentConfig


class IConfigLoader(ABC):
    """Reads and parses YAML configuration files."""

    @abstractmethod
    def read(self) -> dict[str, Any]:
        """Load and parse YAML configuration into a dictionary."""
        pass


class IConfigValidator(ABC):
    """Validates configuration structure and required fields."""

    @abstractmethod
    def validate_root(self, data: dict[str, Any]) -> None:
        """Validate top-level YAML structure."""
        pass

    @abstractmethod
    def validate_problem(self, problem: dict[str, Any]) -> None:
        """Validate 'problem' section fields."""
        pass

    @abstractmethod
    def validate_algorithm(self, algorithm: dict[str, Any], allow_lists: bool) -> None:
        """Validate 'algorithm' section and operator configs."""
        pass


class IConfigExpander(ABC):
    """Expands YAML entries into experiment configurations."""

    @abstractmethod
    def expand(self, data: dict[str, Any]) -> List[ExperimentConfig]:
        """Expand parsed YAML into ExperimentConfig instances."""
        pass


class INameGenerator(ABC):
    """Generates unique experiment identifiers."""

    @abstractmethod
    def generate(self, problem: dict[str, Any], alg: dict[str, Any]) -> str:
        """Generate standardized experiment name."""
        pass


class IConfigService(ABC):
    """Manages loading, validation, and expansion of configurations."""

    @abstractmethod
    def load_all(self) -> List[ExperimentConfig]:
        """Return all validated and expanded experiment configurations."""
        pass


class IResultCollector(ABC):
    """Collects and saves algorithm execution results."""

    @abstractmethod
    def collect_run(self, config_name: str, history: List[Tuple[float, float]]) -> None:
        """Store results of a single algorithm run."""
        pass

    @abstractmethod
    def finalize_config(self, config_name: str, optimal_value: float | None, runs: int) -> None:
        """Aggregate results across runs and save summary."""
        pass


class IStatistics(ABC):
    """Computes summary metrics for experiment results."""

    @abstractmethod
    def compute_mean_error(self, results: List[float], optimum: float | None) -> float:
        """Compute mean relative error across runs."""
        pass

    @abstractmethod
    def best_cost(self, results: List[float]) -> float:
        """Return lowest cost among all runs."""
        pass


class IExperimentRunner(ABC):
    """Executes multiple experiment configurations."""

    @abstractmethod
    def run_all(self, configs: List[ExperimentConfig]) -> None:
        """Run all experiments and delegate results to collector."""
        pass
