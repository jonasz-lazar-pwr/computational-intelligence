from abc import ABC, abstractmethod
from typing import Tuple


class IAlgorithm(ABC):
    """Abstract interface for all optimization algorithms."""

    @abstractmethod
    def initialize_population(self) -> None:
        """Initialize population or starting state."""
        pass

    @abstractmethod
    def evolve(self) -> None:
        """Run main algorithm loop."""
        pass

    @abstractmethod
    def get_best_solution(self) -> Tuple[list[int], float]:
        """Return best solution and its fitness."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset algorithm state (for repeated experiments)."""
        pass
