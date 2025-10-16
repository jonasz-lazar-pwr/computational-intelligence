import random
import time

from typing import Any, Dict, List

from src.core.logger import get_logger
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.factories_interfaces import IOperatorFactory
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class GeneticAlgorithm(IAlgorithm):
    """Genetic Algorithm for TSP or other permutation-based optimization problems."""

    def __init__(  # noqa: PLR0913
        self,
        problem: IProblem,
        operator_factory: IOperatorFactory,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        max_time: float,
        selection_config: dict[str, Any],
        crossover_config: dict[str, Any],
        mutation_config: dict[str, Any],
    ) -> None:
        """Initialize GA parameters and load operators from factory."""
        super().__init__()
        self.problem: IProblem = problem
        self.population_size: int = population_size
        self.crossover_rate: float = crossover_rate
        self.mutation_rate: float = mutation_rate
        self.max_time: float = max_time

        self.selection: ISelection = operator_factory.get_operator("selection", **selection_config)
        self.crossover: ICrossover = operator_factory.get_operator("crossover", **crossover_config)
        self.mutation: IMutation = operator_factory.get_operator("mutation", **mutation_config)

        self.best_solution: List[int] = []
        self.best_cost: float = float("inf")

    def _initialize_population(self) -> List[List[int]]:
        """Generate a random initial population."""
        base = self.problem.get_initial_solution()
        return [random.sample(base, len(base)) for _ in range(self.population_size)]

    def _evaluate_population(self, population: List[List[int]]) -> List[float]:
        """Compute the cost of each individual."""
        return [self.problem.evaluate(ind) for ind in population]

    def _update_best(self, solution: List[int], cost: float) -> None:
        """Store solution if it improves the global best."""
        if cost < self.best_cost:
            self.best_solution = solution[:]
            self.best_cost = cost

    def _replace_population(
        self,
        parents: List[List[int]],
        offspring: List[List[int]],
        parent_costs: List[float],
        offspring_costs: List[float],
    ) -> tuple[List[List[int]], List[float]]:
        """Select top individuals by cost to form next generation."""
        combined = list(zip(parents + offspring, parent_costs + offspring_costs, strict=False))
        combined.sort(key=lambda x: x[1])
        new_population = [x[0] for x in combined[: self.population_size]]
        new_costs = [x[1] for x in combined[: self.population_size]]
        return new_population, new_costs

    def run(self) -> Dict[str, Any]:
        """Execute the main GA optimization loop."""
        start = time.time()
        population = self._initialize_population()
        costs = self._evaluate_population(population)

        best_idx = costs.index(min(costs))
        self._update_best(population[best_idx], costs[best_idx])

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
            population, costs = self._replace_population(
                population, offspring, costs, offspring_costs
            )

            self._update_best(population[0], costs[0])

        elapsed = time.time() - start
        logger.info(f"GA finished: best_cost={self.best_cost:.2f}, time={elapsed:.2f}s")
        return {
            "best_solution": self.best_solution,
            "best_cost": self.best_cost,
            "execution_time": elapsed,
        }
