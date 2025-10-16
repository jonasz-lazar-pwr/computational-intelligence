import random

from typing import Any, List, Tuple

from src.algorithms.base_algorithm import BaseAlgorithm
from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection
from src.interfaces.problems_interfaces import IProblem
from src.operators.registry import (
    build_crossover,
    build_mutation,
    build_selection,
)

logger = get_logger(__name__)


class GeneticAlgorithm(BaseAlgorithm):
    """Genetic Algorithm using DI for selection, crossover, and mutation."""

    def __init__(self, problem: IProblem, config: dict[str, Any]) -> None:
        super().__init__(problem, config)
        self.population_size: int = config.get("population_size", 100)
        self.crossover_rate: float = config.get("crossover_rate", 0.8)
        self.mutation_rate: float = config.get("mutation_rate", 0.1)
        self.elitism: bool = config.get("elitism", True)

        sel_name = config.get("selection", "tournament")
        cx_name = config.get("crossover", "ox")
        mut_name = config.get("mutation", "swap")
        self.selection: ISelection = build_selection(sel_name, **config.get("selection_params", {}))
        self.crossover: ICrossover = build_crossover(cx_name, **config.get("crossover_params", {}))
        self.mutation: IMutation = build_mutation(mut_name, **config.get("mutation_params", {}))

    def initialize_population(self) -> None:
        """Create an initial population of random permutations."""
        n = self.problem.get_dimension()
        self.population = [random.sample(range(n), n) for _ in range(self.population_size)]

    def evolve(self) -> None:
        """Perform one generation."""
        new_pop: List[List[int]] = []
        if self.elitism and self.best_solution is not None:
            new_pop.append(self.best_solution[0])

        # precompute fitness for selection
        fitness = [self.problem.evaluate(ind) for ind in self.population]

        while len(new_pop) < self.population_size:
            p1 = self.selection.select(self.population, fitness)
            p2 = self.selection.select(self.population, fitness)

            if random.random() < self.crossover_rate:
                c1, c2 = self.crossover.crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < self.mutation_rate:
                self.mutation.mutate(c1)
            if random.random() < self.mutation_rate:
                self.mutation.mutate(c2)

            new_pop.extend([c1, c2])

        self.population = new_pop[: self.population_size]

    def get_best_solution(self) -> Tuple[list[int], float]:
        """Return the current best solution and fitness."""
        if self.best_solution is None:
            self.evaluate_generation()
        return self.best_solution
