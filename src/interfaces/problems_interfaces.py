from abc import ABC, abstractmethod
from typing import Any, List, Optional


class IProblem(ABC):
    """Generic interface for all optimization problems."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return number of decision variables."""
        pass

    @abstractmethod
    def evaluate(self, solution: List[int]) -> float:
        """Compute the fitness (cost) for a given solution."""
        pass

    @abstractmethod
    def get_initial_solution(self) -> List[int]:
        """Return a valid initial solution."""
        pass

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Return metadata about the problem (optional)."""
        pass

    @abstractmethod
    def optimal_value(self) -> Optional[float]:
        """Return known optimal objective value (if available)."""
        pass
