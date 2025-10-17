from typing import List, Tuple

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ISuccession

logger = get_logger(__name__)


class SteadyStateSuccession(ISuccession):
    """Implements steady-state succession with partial population replacement."""

    def __init__(self, replacement_rate: float = 0.1) -> None:
        """Initialize with the fraction of individuals to replace each generation."""
        if not 0 < replacement_rate <= 1:
            raise ValueError("Replacement rate must be in (0, 1].")
        self.replacement_rate = replacement_rate

    def replace(
        self,
        parents: List[List[int]],
        offspring: List[List[int]],
        parent_costs: List[float],
        offspring_costs: List[float],
    ) -> Tuple[List[List[int]], List[float]]:
        """Return new population by replacing worst parents with best offspring."""
        population_size = len(parents)
        replace_count = max(1, int(self.replacement_rate * population_size))
        parent_pairs = list(zip(parents, parent_costs, strict=False))
        parent_pairs.sort(key=lambda x: x[1])
        offspring_pairs = list(zip(offspring, offspring_costs, strict=False))
        offspring_pairs.sort(key=lambda x: x[1])
        survivors = (
            parent_pairs[: population_size - replace_count] + offspring_pairs[:replace_count]
        )
        new_population = [x[0] for x in survivors]
        new_costs = [x[1] for x in survivors]
        logger.debug(
            f"Steady-state succession: replaced {replace_count}/{population_size} individuals."
        )
        return new_population, new_costs
