from typing import Any, List

from src.core.logger import get_logger
from src.interfaces.tsp_interfaces import ITSPInstance
from src.problems.base_problem import BaseProblem

logger = get_logger(__name__)


class TSPProblem(BaseProblem):
    """Adapter exposing ITSPInstance to algorithms."""

    def __init__(self, instance: ITSPInstance) -> None:
        """Initialize TSPProblem with given TSP instance."""
        super().__init__(instance.name or "TSP", instance.dimension or 0)
        self.instance = instance
        if not instance.has_loaded:
            instance.load_distance_matrix()
        logger.debug(f"TSPProblem initialized for {self.instance.name}")

    def evaluate(self, solution: List[int]) -> float:
        """Calculate total tour distance for the given solution."""
        dist = self.instance.get_distance_matrix()
        if not dist:
            raise RuntimeError("Distance matrix not loaded.")
        total = sum(
            dist[solution[i]][solution[(i + 1) % len(solution)]] for i in range(len(solution))
        )
        return float(total)

    def get_initial_solution(self) -> List[int]:
        """Return default sequential tour."""
        return list(range(self.get_dimension()))

    def optimal_value(self) -> float | None:
        """Return known optimal value if available."""
        return self.instance.optimal_result

    def info(self) -> dict[str, Any]:
        """Return combined base and TSP-specific metadata."""
        data = super().info()
        data.update(
            {
                "edge_weight_type": self.instance.edge_weight_type,
                "optimal_result": self.optimal_value(),
            }
        )
        return data
