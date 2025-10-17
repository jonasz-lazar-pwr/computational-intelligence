import random

from typing import List

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ISelection

logger = get_logger(__name__)


class RouletteSelection(ISelection):
    """Implements roulette wheel (fitness-proportionate) selection."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        """Initialize the selector with epsilon to avoid division by zero."""
        self.epsilon = epsilon

    def select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Return one individual selected proportionally to its fitness."""
        fitness = [1.0 / (c + self.epsilon) for c in costs]
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        r = random.uniform(0, 1)
        cumulative = 0.0
        for individual, prob in zip(population, probabilities, strict=False):
            cumulative += prob
            if r <= cumulative:
                logger.debug(f"Roulette selection: r={r:.4f}, selected_prob={prob:.4f}")
                return individual
        logger.debug(f"Roulette selection: r={r:.4f}, fallback to last individual.")
        return population[-1]
