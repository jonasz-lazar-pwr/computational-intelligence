from abc import ABC, abstractmethod
from typing import Any

from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection, ISuccession
from src.interfaces.problems_interfaces import IProblem


class IAlgorithmFactory(ABC):
    """Defines interface for creating algorithm instances."""

    @abstractmethod
    def build(self, name: str, **config: Any) -> IAlgorithm:
        """Return an algorithm instance by name."""
        pass


class IProblemFactory(ABC):
    """Defines interface for creating problem instances."""

    @abstractmethod
    def build(self, name: str, **config: Any) -> IProblem:
        """Return a problem instance by name."""
        pass


class IOperatorFactory(ABC):
    """Defines interface for creating operator instances."""

    @abstractmethod
    def get_operator(
        self, category: str, name: str, **config: Any
    ) -> ISelection | ICrossover | IMutation | ISuccession:
        """Return an operator instance for the given category and name."""
        pass
