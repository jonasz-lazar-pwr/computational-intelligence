import time

from abc import ABC
from typing import List

import pytest

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.factories.operator_factory import OperatorFactory
from src.interfaces.problems_interfaces import IProblem


class DummyProblem(IProblem, ABC):
    """Dummy implementation of IProblem for GA tests."""

    def __init__(self) -> None:
        """Initialize dummy problem with fixed dimension."""
        self._dimension = 4

    def get_initial_solution(self) -> List[int]:
        """Return base permutation."""
        return [0, 1, 2, 3]

    def evaluate(self, ind: List[int]) -> float:
        """Return sum of individual elements."""
        return float(sum(ind))

    def get_dimension(self) -> int:
        """Return dimension size."""
        return self._dimension

    def optimal_value(self) -> float | None:
        """Return dummy optimal value (unused in GA)."""
        return None

    def info(self) -> dict:
        """Return minimal info dictionary."""
        return {"name": "DummyProblem", "dimension": self._dimension}


@pytest.fixture(scope="session")
def mock_problem() -> DummyProblem:
    """Provide dummy problem fixture."""
    return DummyProblem()


@pytest.fixture(scope="session")
def operator_factory() -> OperatorFactory:
    """Provide operator factory fixture."""
    return OperatorFactory()


@pytest.fixture()
def genetic_algorithm(
    mock_problem: DummyProblem, operator_factory: OperatorFactory
) -> GeneticAlgorithm:
    """Return configured GA instance."""
    return GeneticAlgorithm(
        problem=mock_problem,
        operator_factory=operator_factory,
        population_size=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        max_time=0.05,
        selection_config={"name": "tournament", "tournament_size": 2},
        crossover_config={"name": "ox"},
        mutation_config={"name": "insert"},
    )


def test_initialize_population(genetic_algorithm: GeneticAlgorithm):
    """Check random population initialization."""
    population = genetic_algorithm._initialize_population()
    assert isinstance(population, list)
    assert len(population) == genetic_algorithm.population_size
    assert all(isinstance(ind, list) for ind in population)
    assert all(len(ind) == genetic_algorithm.problem.get_dimension() for ind in population)


def test_evaluate_population(genetic_algorithm: GeneticAlgorithm):
    """Check population evaluation returns numeric values."""
    population = [[0, 1, 2, 3], [3, 2, 1, 0]]
    costs = genetic_algorithm._evaluate_population(population)
    assert all(isinstance(c, float) for c in costs)
    assert costs == [6.0, 6.0]


def test_update_best_improves_only(genetic_algorithm: GeneticAlgorithm):
    """Check best update occurs only for improvement."""
    ga = genetic_algorithm
    ga._update_best([1, 2, 3, 4], 10.0)
    assert ga.best_cost == 10.0
    ga._update_best([2, 3, 4, 5], 15.0)
    assert ga.best_cost == 10.0
    ga._update_best([0, 1, 2, 3], 5.0)
    assert ga.best_cost == 5.0


def test_replace_population_selects_best(genetic_algorithm: GeneticAlgorithm):
    """Check replacement keeps best individuals."""
    parents = [[0, 1, 2, 3], [3, 2, 1, 0]]
    offspring = [[1, 0, 2, 3], [2, 1, 3, 0]]
    parent_costs = [100.0, 50.0]
    offspring_costs = [20.0, 80.0]
    new_pop, new_costs = genetic_algorithm._replace_population(
        parents, offspring, parent_costs, offspring_costs
    )
    assert len(new_pop) == genetic_algorithm.population_size
    assert new_costs == sorted(new_costs)
    assert min(new_costs) == 20.0
    assert max(new_costs) <= 100.0


def test_run_executes_and_returns_results(genetic_algorithm: GeneticAlgorithm):
    """Check GA run returns valid results."""
    result = genetic_algorithm.run()
    assert isinstance(result, dict)
    assert {"best_solution", "best_cost", "execution_time"} <= result.keys()
    assert isinstance(result["best_cost"], float)
    assert isinstance(result["execution_time"], float)
    assert len(result["best_solution"]) == genetic_algorithm.problem.get_dimension()


def test_run_stops_within_time_limit(genetic_algorithm: GeneticAlgorithm):
    """Check GA stops within max_time limit."""
    start = time.time()
    genetic_algorithm.max_time = 0.05
    genetic_algorithm.run()
    elapsed = time.time() - start
    assert elapsed < 0.5


def test_internal_randomness_does_not_break(genetic_algorithm: GeneticAlgorithm, monkeypatch):
    """Check algorithm handles crossover and mutation safely."""
    ga = genetic_algorithm
    monkeypatch.setattr("random.random", lambda: 0.0)
    population = ga._initialize_population()
    costs = ga._evaluate_population(population)
    ga.selection.select(population, costs)
    ga.crossover.crossover(population[0], population[1])
    ga.mutation.mutate(population[0])
    result = ga.run()
    assert result["best_cost"] >= 0.0
    assert isinstance(result["best_solution"], list)
