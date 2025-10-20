import random
import time

from typing import Any, Dict, List

from src.core.logger import get_logger
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.operators_interfaces import (
    ICrossover,
    IMutation,
    ISelection,
    ISuccession,
)
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class GeneticAlgorithm(IAlgorithm):
    """Genetic algorithm for permutation-based optimization with stagnation-based stop."""

    def __init__(  # noqa: PLR0913
        self,
        problem: IProblem,
        selection: ISelection,
        crossover: ICrossover,
        mutation: IMutation,
        succession: ISuccession,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        max_time: float,
        seed: int | None = None,
    ) -> None:
        """Initialize the algorithm with operators, parameters, and optional seed."""
        super().__init__()
        self.problem = problem
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.succession = succession
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_time = max_time

        self.best_cost: float = float("inf")
        self.history: list[tuple[float, float]] = []
        self._rng = random.Random(seed)

        self._no_improvement_limit = 3.0
        self._last_improvement_time: float | None = None

        if seed is not None:
            logger.debug(f"GeneticAlgorithm initialized with seed={seed}")

    def _random(self) -> float:
        """Return a random float using the local RNG."""
        return self._rng.random()

    def _sample(self, seq: List[int], k: int) -> List[int]:
        """Return a random sample of elements using the local RNG."""
        return self._rng.sample(seq, k)

    def _initialize_population(self) -> List[List[int]]:
        """Generate the initial random population."""
        base = self.problem.get_initial_solution()
        return [self._sample(base, len(base)) for _ in range(self.population_size)]

    def _evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Evaluate all individuals and return their costs."""
        return [self.problem.evaluate(ind) for ind in population]

    def _update_best(self, cost: float, now: float) -> None:
        """Update the best cost found so far and reset stagnation timer."""
        if cost < self.best_cost:
            self.best_cost = cost
            self._last_improvement_time = now

    def run(self) -> Dict[str, Any]:
        """Run the genetic algorithm until time or stagnation limit is reached."""
        start = time.time()
        self._last_improvement_time = start

        population = self._initialize_population()
        costs = self._evaluate_population(population)
        self._update_best(min(costs), start)

        while True:
            now = time.time()
            elapsed = now - start
            stagnation = now - (self._last_improvement_time or start)

            # Stop condition
            if elapsed >= self.max_time or stagnation >= self._no_improvement_limit:
                break

            offspring: List[List[int]] = []
            while len(offspring) < self.population_size:
                p1 = self.selection.select(population, costs)
                p2 = self.selection.select(population, costs)

                # Crossover
                if self._random() < self.crossover_rate:
                    c1, c2 = self.crossover.crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # Mutation
                if self._random() < self.mutation_rate:
                    self.mutation.mutate(c1)
                if self._random() < self.mutation_rate:
                    self.mutation.mutate(c2)

                offspring.extend([c1, c2])

            # Evaluate and replace
            offspring_costs = self._evaluate_population(offspring)
            population, costs = self.succession.replace(
                population, offspring, costs, offspring_costs
            )

            current_best = min(costs)
            self._update_best(current_best, now)

            elapsed_ms = (time.time() - start) * 1000
            self.history.append((elapsed_ms, self.best_cost))

        logger.info(
            f"GA finished: best_cost={self.best_cost:.2f}, "
            f"samples={len(self.history)}, "
            f"elapsed={elapsed:.2f}s, stagnation={stagnation:.2f}s"
        )

        return {"history": self.history}
