from typing import List, Optional, Tuple

from src.interfaces.operators_interfaces import ICrossover


class CycleCrossover(ICrossover):
    """Implements the Cycle Crossover (CX) operator."""

    def crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        """Return two offspring generated using cycle crossover."""
        size = len(p1)

        def cx(parent1: List[int], parent2: List[int]) -> List[int]:
            """Return one offspring from a CX operation."""
            child: List[Optional[int]] = [None] * size
            remaining = set(range(size))

            while remaining:
                start = remaining.pop()
                idx = start
                cycle_indices = [idx]
                value = parent2[idx]
                idx = parent1.index(value)
                while idx != start:
                    cycle_indices.append(idx)
                    remaining.remove(idx)
                    value = parent2[idx]
                    idx = parent1.index(value)
                for i in cycle_indices:
                    child[i] = parent1[i]
            for i in range(size):
                if child[i] is None:
                    child[i] = parent2[i]
            return [x for x in child if x is not None]

        return cx(p1, p2), cx(p2, p1)
