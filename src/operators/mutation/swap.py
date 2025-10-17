import random

from typing import List

from src.interfaces.operators_interfaces import IMutation

_MIN_GENES = 2


class SwapMutation(IMutation):
    """Implements the swap mutation operator."""

    def mutate(self, individual: List[int]) -> None:
        """Swap two random genes in the given chromosome."""
        if len(individual) < _MIN_GENES:
            return
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
