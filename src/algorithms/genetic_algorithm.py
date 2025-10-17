import random
import time

from typing import Any, Dict, List

from src.core.logger import get_logger
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection, ISuccession
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class GeneticAlgorithm(IAlgorithm):
    """Implements a genetic algorithm for permutation-based optimization."""

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
    ) -> None:
        """Initialize the genetic algorithm with operators and parameters."""
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
        self.best_solution: List[int] = []
        self.best_cost: float = float("inf")

    def _initialize_population(self) -> List[List[int]]:
        """Generate the initial random population."""
        base = self.problem.get_initial_solution()
        return [random.sample(base, len(base)) for _ in range(self.population_size)]

    def _evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Compute fitness (cost) for each individual."""
        return [self.problem.evaluate(ind) for ind in population]

    def _update_best(self, solution: List[int], cost: float) -> None:
        """Update the current best solution if a better one is found."""
        if cost < self.best_cost:
            self.best_solution = solution[:]
            self.best_cost = cost

    def run(self) -> Dict[str, Any]:
        """Execute the main genetic algorithm loop until the time limit is reached."""
        start = time.time()
        population = self._initialize_population()
        costs = self._evaluate_population(population)
        self._update_best(population[costs.index(min(costs))], min(costs))

        while time.time() - start < self.max_time:
            offspring: List[List[int]] = []
            while len(offspring) < self.population_size:
                p1 = self.selection.select(population, costs)
                p2 = self.selection.select(population, costs)

                if random.random() < self.crossover_rate:
                    c1, c2 = self.crossover.crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                if random.random() < self.mutation_rate:
                    self.mutation.mutate(c1)
                if random.random() < self.mutation_rate:
                    self.mutation.mutate(c2)

                offspring.extend([c1, c2])

            offspring_costs = self._evaluate_population(offspring)
            population, costs = self.succession.replace(
                population, offspring, costs, offspring_costs
            )
            best_idx = costs.index(min(costs))
            self._update_best(population[best_idx], costs[best_idx])

        elapsed = time.time() - start
        logger.info(f"GA finished: best_cost={self.best_cost:.2f}, time={elapsed:.2f}s")
        return {
            "best_solution": self.best_solution,
            "best_cost": self.best_cost,
            "execution_time": elapsed,
        }
