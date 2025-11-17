from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import pandas as pd

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
    def collect_run(self, config_name: str, best_cost: float) -> None:
        """Store results of a single algorithm run."""
        pass

    @abstractmethod
    def finalize_config(self, config_name: str, optimal_value: float | None, runs: int) -> None:
        """Aggregate results across runs and save results."""
        pass


class IStatistics(ABC):
    """Computes results metrics for experiment results."""

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


class IResultParser(ABC):
    """Interface for parsing result files."""

    @abstractmethod
    def load(self) -> None:
        """"""
        pass

    @abstractmethod
    def parse(self) -> None:
        """"""
        pass

    @abstractmethod
    def export_csv(self, output_path: Path) -> None:
        """"""
        pass

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """"""
        pass


class ILatexTableGenerator(ABC):
    """Interface for generating LaTeX tables from experiment data."""

    @abstractmethod
    def generate(self, csv_path: Path, output_path: Path, top_n: int) -> None:
        """Generate LaTeX table from CSV and export to file."""
        pass


class IComparisonPlotGenerator(ABC):
    """Interface for generating comparison plots from CSV results."""

    @abstractmethod
    def generate_selection_by_population(self, csv_path: Path, output_path: Path) -> None:
        """Bar chart: selection x population -> mean error [%]."""
        pass

    @abstractmethod
    def generate_crossover_by_succession(self, csv_path: Path, output_path: Path) -> None:
        """Bar chart: (crossover,param) x succession -> mean error [%]."""
        pass

    @abstractmethod
    def generate_mutation_by_selection(self, csv_path: Path, output_path: Path) -> None:
        """Bar chart: (mutation,param) x selection -> mean error [%]."""
        pass

    @abstractmethod
    def generate_succession_vs_selection_heatmap(self, csv_path: Path, output_path: Path) -> None:
        """Bar chart: (succession,param) x selection -> mean error [%]."""
        pass
