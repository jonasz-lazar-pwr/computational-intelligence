from typing import List, Tuple

from src.interfaces.operators_interfaces import ICrossover


class OrderCrossover(ICrossover):
    """Order Crossover (OX)."""

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Return two offspring created by OX."""
        n = len(p1)
        a, b = sorted(__import__("random").sample(range(n), 2))
        c1 = [None] * n
        c2 = [None] * n
        c1[a:b] = p1[a:b]
        c2[a:b] = p2[a:b]
        fill1 = [x for x in p2 if x not in c1]
        fill2 = [x for x in p1 if x not in c2]
        i1 = i2 = 0
        for i in range(n):
            if c1[i] is None:
                c1[i] = fill1[i1]
                i1 += 1
            if c2[i] is None:
                c2[i] = fill2[i2]
                i2 += 1
        return c1, c2
