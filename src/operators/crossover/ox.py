import random

from typing import List, Optional, Tuple

from src.interfaces.operators_interfaces import ICrossover


class OrderCrossover(ICrossover):
    """Order Crossover (OX) preserving relative order of elements."""

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Generate two offspring from two parents using order crossover."""
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))

        def ox(parent1: List[int], parent2: List[int]) -> List[int]:
            """Perform OX between two parents."""
            child: List[Optional[int]] = [None] * size
            child[a:b] = parent1[a:b]
            fill = [x for x in parent2 if x not in child]
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            return [x for x in child if x is not None]

        return ox(p1, p2), ox(p2, p1)
