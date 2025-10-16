import random

from typing import List

from src.interfaces.operators_interfaces import IMutation


class InsertMutation(IMutation):
    """Insert mutation for permutation chromosomes."""

    def mutate(self, individual: List[int]) -> None:
        """Move one element to a new random position."""
        i, j = random.sample(range(len(individual)), 2)
        gene = individual.pop(i)
        individual.insert(j, gene)
