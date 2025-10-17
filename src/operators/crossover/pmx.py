import random

from typing import List, Optional, Tuple

from src.interfaces.operators_interfaces import ICrossover


class PartiallyMappedCrossover(ICrossover):
    """Implements the Partially Mapped Crossover (PMX) operator."""

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Return two offspring generated using PMX crossover."""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))

        def pmx(parent1: List[int], parent2: List[int]) -> List[int]:
            """Return one offspring from a PMX operation."""
            child: List[Optional[int]] = [None] * size
            child[a:b] = parent1[a:b]
            for i in range(a, b):
                if parent2[i] not in child:
                    pos = i
                    val = parent2[i]
                    while True:
                        idx = parent2.index(parent1[pos])
                        if child[idx] is None:
                            child[idx] = val
                            break
                        pos = idx
            for i in range(size):
                if child[i] is None:
                    child[i] = parent2[i]
            return [int(x) for x in child if x is not None]

        return pmx(p1, p2), pmx(p2, p1)
