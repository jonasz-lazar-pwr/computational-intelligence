import random

from typing import List

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ISelection

logger = get_logger(__name__)


class RankSelection(ISelection):
    """Implements rank-based selection."""

    def select(self, population: List[List[int]], costs: List[float]) -> List[int]:
        """Return one individual selected using rank-based probability."""
        ranked = sorted(zip(population, costs, strict=False), key=lambda x: x[1])
        n = len(ranked)
        ranks = list(range(n, 0, -1))
        total = sum(ranks)
        probabilities = [r / total for r in ranks]
        r = random.uniform(0, 1)
        cumulative = 0.0
        for (individual, _), p in zip(ranked, probabilities, strict=False):
            cumulative += p
            if r <= cumulative:
                logger.debug(f"Rank selection: r={r:.4f}, selected_rank_prob={p:.4f}")
                return individual
        logger.debug(f"Rank selection: r={r:.4f}, fallback to worst-ranked individual.")
        return ranked[-1][0]
