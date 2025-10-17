from typing import List, Tuple

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ISuccession

logger = get_logger(__name__)


class ElitistSuccession(ISuccession):
    """Implements elitist succession preserving top-performing parents."""

    def __init__(self, elite_rate: float = 0.1) -> None:
        """Initialize with the fraction of parents to preserve as elites."""
        if not 0 < elite_rate <= 1:
            raise ValueError("Elite rate must be in the range (0, 1].")
        self.elite_rate = elite_rate

    def replace(
        self,
        parents: List[List[int]],
        offspring: List[List[int]],
        parent_costs: List[float],
        offspring_costs: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        """Return new population preserving elites and best offspring."""
        population_size = len(parents)
        elite_count = max(1, int(self.elite_rate * population_size))
        parent_pairs = list(zip(parents, parent_costs, strict=False))
        parent_pairs.sort(key=lambda x: x[1])
        elites = parent_pairs[:elite_count]
        offspring_pairs = list(zip(offspring, offspring_costs, strict=False))
        offspring_pairs.sort(key=lambda x: x[1])
        survivors = elites + offspring_pairs[: population_size - elite_count]
        new_population = [x[0] for x in survivors]
        new_costs = [x[1] for x in survivors]
        logger.debug(
            f"Elitist succession: preserved {elite_count}/{population_size} parents ({self.elite_rate:.0%})."
        )
        return new_population, new_costs
