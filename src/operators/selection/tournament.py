import random

from typing import List

from src.interfaces.operators_interfaces import ISelection


class TournamentSelection(ISelection):
    """k-tournament selection."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

    def select(self, population: List[List[int]], fitness: list[float]) -> List[int]:
        """Return the best of k random individuals."""
        idxs = random.sample(range(len(population)), self.k)
        best_idx = min(idxs, key=lambda i: fitness[i])
        return population[best_idx][:]
