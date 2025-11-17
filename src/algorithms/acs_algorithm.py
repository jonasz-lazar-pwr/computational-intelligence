import random
import time

from typing import Any, Dict, List, Tuple

from src.core.logger import get_logger
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class ACSAlgorithm(IAlgorithm):
    """Ant Colony System algorithm implementation."""

    def __init__(  # noqa: PLR0913
        self,
        problem: IProblem,
        num_ants: int,
        alpha: float,
        beta: float,
        rho: float,
        phi: float,
        q0: float,
        max_time: float,
        seed: int | None = None,
    ) -> None:
        """Initialize parameters, RNG and internal structures."""
        super().__init__()
        self.problem = problem
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.phi = phi
        self.q0 = q0
        self.max_time = max_time

        self.best_cost: float = float("inf")
        self.history: List[Tuple[float, float]] = []
        self._rng = random.Random(seed)

        self._pheromone: List[List[float]] = []
        self._heuristic: List[List[float]] = []

        self._no_improvement_limit = 2.0
        self._last_improvement_time: float | None = None

        logger.debug(
            f"ACS initialized: ants={num_ants}, alpha={alpha}, beta={beta}, "
            f"rho={rho}, phi={phi}, q0={q0}, seed={seed}"
        )

    def _update_best(self, cost: float, now: float) -> None:
        """Update global best and stagnation timer."""
        if cost < self.best_cost:
            self.best_cost = cost
            self._last_improvement_time = now

    def _initialize_pheromone_and_heuristic(self) -> None:
        """Initialize pheromone and heuristic matrices."""
        n = self.problem.get_dimension()
        self._pheromone = [[1.0 for _ in range(n)] for _ in range(n)]

        self._heuristic = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                else:
                    d = self.problem.get_distance(i, j)
                    row.append(1.0 / d if d > 0 else 0.0)
            self._heuristic.append(row)

        logger.debug("ACS pheromone and heuristic matrices initialized.")

    def _choose_next_city(self, current: int, unvisited: List[int]) -> int:
        """Select next city using ACS decision rule."""
        q = self._rng.random()

        if q <= self.q0:
            return max(
                unvisited,
                key=lambda j: (self._pheromone[current][j] ** self.alpha)
                * (self._heuristic[current][j] ** self.beta),
            )

        weights = [
            (self._pheromone[current][j] ** self.alpha) * (self._heuristic[current][j] ** self.beta)
            for j in unvisited
        ]

        total = sum(weights)
        if total == 0:
            return self._rng.choice(unvisited)

        r = self._rng.random()
        cum = 0.0
        for j, w in zip(unvisited, weights, strict=False):
            cum += w / total
            if r <= cum:
                return j

        return unvisited[-1]

    def _local_update(self, i: int, j: int) -> None:
        """Apply local pheromone update."""
        self._pheromone[i][j] = (1 - self.phi) * self._pheromone[i][j] + self.phi * 1.0

    def _global_update(self, route: List[int], cost: float) -> None:
        """Apply global pheromone update."""
        deposit = 1.0 / cost
        for a, b in zip(route, [*route[1:], route[0]], strict=False):
            updated = (1 - self.rho) * self._pheromone[a][b] + self.rho * deposit
            self._pheromone[a][b] = updated
            self._pheromone[b][a] = updated

    def _build_route(self) -> List[int]:
        """Construct a route using ACS rules."""
        n = self.problem.get_dimension()
        start = self._rng.randrange(n)
        route = [start]

        unvisited = set(range(n))
        unvisited.remove(start)

        current = start
        while unvisited:
            nxt = self._choose_next_city(current, list(unvisited))
            self._local_update(current, nxt)

            route.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        return route

    def run(self) -> Dict[str, Any]:
        """Execute ACS until time or stagnation limit."""
        self._initialize_pheromone_and_heuristic()

        start = time.time()
        self._last_improvement_time = start

        while True:
            now = time.time()
            elapsed = now - start
            stagnation = now - (self._last_improvement_time or start)

            if elapsed >= self.max_time or stagnation >= self._no_improvement_limit:
                break

            routes = []
            costs = []

            for _ in range(self.num_ants):
                route = self._build_route()
                cost = self.problem.evaluate(route)
                routes.append(route)
                costs.append(cost)

            idx = min(range(len(costs)), key=lambda k: costs[k])
            iteration_best_route = routes[idx]
            iteration_best_cost = costs[idx]

            self._update_best(iteration_best_cost, now)
            self._global_update(iteration_best_route, iteration_best_cost)

            elapsed_ms = (time.time() - start) * 1000
            self.history.append((elapsed_ms, self.best_cost))

        logger.info(
            f"ACS finished: best_cost={self.best_cost:.2f}, "
            f"samples={len(self.history)}, "
            f"elapsed={elapsed:.2f}s, stagnation={stagnation:.2f}s"
        )

        return {"history": self.history, "best_cost": self.best_cost}
