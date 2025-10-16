import random

from typing import List

from src.interfaces.operators_interfaces import ISelection


class TournamentSelection(ISelection):
    """Tournament selection operator."""

    def __init__(self, tournament_size: int) -> None:
        """Initialize with the given tournament size."""
        self.k: int = tournament_size

    def select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Choose the best individual from a random subset of the population."""
        participants = random.sample(list(zip(population, costs, strict=False)), self.k)
        winner = min(participants, key=lambda x: x[1])
        return winner[0]
