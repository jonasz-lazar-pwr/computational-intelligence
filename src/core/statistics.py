from typing import List

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IStatistics

logger = get_logger(__name__)


class Statistics(IStatistics):
    """Computes mean relative error and best cost across runs."""

    def compute_mean_error(self, results: List[float], optimum: float | None) -> float:
        """Compute mean relative error for given results."""
        if not results:
            logger.warning("No results provided for mean error computation.")
            return 0.0
        if optimum is None or optimum == 0:
            logger.debug("Optimum not provided or zero. Returning mean cost instead.")
            return sum(results) / len(results)
        relative_errors = [(r - optimum) / optimum for r in results]
        mean_error = sum(relative_errors) / len(relative_errors)
        logger.debug(
            f"Computed mean relative error: {mean_error:.6f} "
            f"from {len(results)} runs (optimum={optimum:.3f})."
        )
        return mean_error

    def best_cost(self, results: List[float]) -> float:
        """Return minimum cost among all runs."""
        if not results:
            logger.warning("No results provided for best cost computation.")
            return float("inf")
        best = min(results)
        logger.debug(f"Best cost found: {best:.6f} among {len(results)} runs.")
        return best
