import random

from typing import List

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ISelection

logger = get_logger(__name__)


class TournamentSelection(ISelection):
    """Implements tournament-based selection."""

    def __init__(self, rate: float = 0.1) -> None:
        """Initialize tournament size as a fraction of population size."""
        if not (0 < rate <= 1):
            raise ValueError("Tournament rate must be in (0, 1].")
        self.rate = rate

    def select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Return the best individual among a random subset of the population."""
        k = max(2, int(len(population) * self.rate))
        participants = random.sample(list(zip(population, costs, strict=False)), k)
        winner = min(participants, key=lambda x: x[1])
        logger.debug(f"Tournament selection: k={k}, winner_cost={winner[1]:.2f}")
        return winner[0]
