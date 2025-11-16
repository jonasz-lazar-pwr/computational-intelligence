from abc import ABC, abstractmethod
from typing import Any, List


class IProblem(ABC):
    """Abstract interface for all optimization problems."""

    @abstractmethod
    def evaluate(self, solution: List[int]) -> float:
        """Evaluate the quality or cost of a given solution."""
        pass

    @abstractmethod
    def get_initial_solution(self) -> List[int]:
        """Return an initial candidate solution."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality or size of the problem."""
        pass

    @abstractmethod
    def get_distance(self, i: int, j: int) -> float:
        """Return distance or cost between element i and j."""
        pass

    @abstractmethod
    def optimal_value(self) -> float | None:
        """Return known optimal value if available."""
        pass

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Return metadata about the problem instance."""
        pass
