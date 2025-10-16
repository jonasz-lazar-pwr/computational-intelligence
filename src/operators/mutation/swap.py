import random

from typing import List

from src.interfaces.operators_interfaces import IMutation


class SwapMutation(IMutation):
    """Swap two random positions."""

    def mutate(self, individual: List[int]) -> None:
        """Swap two genes in-place."""
        a, b = random.sample(range(len(individual)), 2)
        individual[a], individual[b] = individual[b], individual[a]
