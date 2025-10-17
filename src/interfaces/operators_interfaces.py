from abc import ABC, abstractmethod
from typing import List, Tuple


class ISelection(ABC):
    """Defines interface for selection operators."""

    @abstractmethod
    def select(self, population: List[List[int]], fitness: List[float]) -> List[int]:
        """Return one selected individual from the population."""
        pass


class ICrossover(ABC):
    """Defines interface for crossover operators."""

    @abstractmethod
    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Return two offspring generated from parent crossover."""
        pass


class IMutation(ABC):
    """Defines interface for mutation operators."""

    @abstractmethod
    def mutate(self, individual: List[int]) -> None:
        """Mutate an individual in-place."""
        pass


class ISuccession(ABC):
    """Defines interface for succession (replacement) strategies."""

    @abstractmethod
    def replace(
        self,
        parents: List[List[int]],
        offspring: List[List[int]],
        parent_costs: List[float],
        offspring_costs: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        """Replace part or all of the population according to strategy."""
        pass
