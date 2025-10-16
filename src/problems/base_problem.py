from abc import ABC
from typing import Any, List, Optional

from src.core.logger import get_logger
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class BaseProblem(IProblem, ABC):
    """Base class for all optimization problems."""

    def __init__(self, name: str, dimension: int) -> None:
        """Initialize problem name and dimension."""
        self.name = name
        self.dimension = dimension
        logger.debug(f"Initialized BaseProblem(name={self.name}, dimension={self.dimension})")

    def evaluate(self, solution: List[int]) -> float:
        """Raise error if subclass does not override evaluation."""
        raise NotImplementedError("BaseProblem cannot evaluate directly.")

    def get_dimension(self) -> int:
        """Return problem dimension."""
        return self.dimension

    def get_initial_solution(self) -> List[int]:
        """Return sequential initial solution."""
        return list(range(self.dimension))

    def info(self) -> dict[str, Any]:
        """Return problem metadata."""
        return {"name": self.name, "dimension": self.dimension}

    def optimal_value(self) -> Optional[float]:
        """Return known optimal value or None."""
        return None
