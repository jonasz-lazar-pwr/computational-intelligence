from abc import ABC, abstractmethod
from typing import List, Tuple


class ISelection(ABC):
    """Select parents from a population."""

    @abstractmethod
    def select(self, population: List[List[int]], fitness: list[float]) -> List[int]:
        """Return one selected individual."""
        pass


class ICrossover(ABC):
    """Produce offspring from two parents."""

    @abstractmethod
    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Return two offspring."""
        pass


class IMutation(ABC):
    """Mutate an individual."""

    @abstractmethod
    def mutate(self, individual: List[int]) -> None:
        """Mutate in-place."""
        pass
