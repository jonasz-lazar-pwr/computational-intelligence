from abc import ABC, abstractmethod
from typing import Any

from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection
from src.interfaces.problems_interfaces import IProblem


class IAlgorithmFactory(ABC):
    """Interface for algorithm factory."""

    @abstractmethod
    def build(self, algorithm_type: str, **kw: Any) -> IAlgorithm:
        """Build and return an algorithm instance."""
        pass


class IProblemFactory(ABC):
    """Interface for problem factory."""

    @abstractmethod
    def build(self, problem_type: str, **kw: Any) -> IProblem:
        """Build and return a problem instance."""
        pass


class IOperatorFactory(ABC):
    """Interface for operator factory."""

    @abstractmethod
    def get_operator(
        self, category: str, name: str, **kw: Any
    ) -> ISelection | ICrossover | IMutation:
        """Return operator instance based on type and name."""
        pass
