from typing import Any, List

from src.core.logger import get_logger
from src.interfaces.problems_interfaces import IProblem
from src.interfaces.tsp_interfaces import ITSPInstance

logger = get_logger(__name__)


class TSPProblem(IProblem):
    """TSP problem implementation compatible with the generic IProblem interface."""

    def __init__(self, instance: ITSPInstance) -> None:
        """Initialize with a TSP instance."""
        self.instance = instance
        instance.load_metadata()
        if not instance.has_loaded:
            instance.load_distance_matrix()
        self.name = instance.name or "TSP"
        self.dimension = instance.dimension or 0
        logger.debug(f"TSPProblem initialized for {self.instance.name}")

    def get_dimension(self) -> int:
        """Return problem dimension."""
        return self.instance.dimension

    def evaluate(self, solution: List[int]) -> float:
        """Compute the total travel cost of a given tour."""
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
        """Return metadata about the TSP instance."""
        return {
            "name": self.instance.name,
            "dimension": self.instance.dimension,
            "edge_weight_type": self.instance.edge_weight_type,
            "optimal_result": self.optimal_value(),
        }
