import time

from typing import List

import pytest

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.factories.operator_factory import OperatorFactory
from src.interfaces.problems_interfaces import IProblem


class DummyProblem(IProblem):
    """Minimal dummy problem for GA testing."""

    def __init__(self) -> None:
        self._dimension = 4

    def get_initial_solution(self) -> List[int]:
        return [0, 1, 2, 3]

    def evaluate(self, ind: List[int]) -> float:
        return float(sum(ind))

    def get_dimension(self) -> int:
        return self._dimension

    def optimal_value(self) -> float | None:
        return None

    def info(self) -> dict:
        return {"name": "DummyProblem", "dimension": self._dimension}


@pytest.fixture(scope="session")
def mock_problem() -> DummyProblem:
    return DummyProblem()


@pytest.fixture(scope="session")
def operator_factory() -> OperatorFactory:
    return OperatorFactory()


@pytest.fixture()
def genetic_algorithm(
    mock_problem: DummyProblem, operator_factory: OperatorFactory
) -> GeneticAlgorithm:
    selection = operator_factory.get_operator("selection", "tournament", rate=0.5)
    crossover = operator_factory.get_operator("crossover", "ox")
    mutation = operator_factory.get_operator("mutation", "insert")
    succession = operator_factory.get_operator("succession", "elitist", elite_rate=0.5)

    return GeneticAlgorithm(
        problem=mock_problem,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        succession=succession,
        population_size=4,
        crossover_rate=0.9,
        mutation_rate=0.1,
        max_time=0.05,
    )


def test_initialize_population(genetic_algorithm: GeneticAlgorithm):
    pop = genetic_algorithm._initialize_population()
    assert isinstance(pop, list)
    assert len(pop) == genetic_algorithm.population_size
    assert all(len(ind) == genetic_algorithm.problem.get_dimension() for ind in pop)


def test_evaluate_population(genetic_algorithm: GeneticAlgorithm):
    pop = [[0, 1, 2, 3], [3, 2, 1, 0]]
    costs = genetic_algorithm._evaluate_population(pop)
    assert all(isinstance(c, float) for c in costs)
    assert costs == [6.0, 6.0]


def test_update_best_improves_only(genetic_algorithm: GeneticAlgorithm):
    ga = genetic_algorithm
    ga._update_best(10.0)
    assert ga.best_cost == 10.0
    ga._update_best(15.0)
    assert ga.best_cost == 10.0
    ga._update_best(5.0)
    assert ga.best_cost == 5.0


def test_run_executes_and_returns_results(genetic_algorithm: GeneticAlgorithm):
    result = genetic_algorithm.run()
    assert isinstance(result, dict)
    assert "history" in result
    assert isinstance(result["history"], list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in result["history"])
    assert all(isinstance(t[0], float) and isinstance(t[1], float) for t in result["history"])


def test_run_stops_within_time_limit(genetic_algorithm: GeneticAlgorithm):
    start = time.time()
    genetic_algorithm.max_time = 0.05
    genetic_algorithm.run()
    elapsed = time.time() - start
    assert elapsed < 0.5


def test_internal_randomness_does_not_break(genetic_algorithm: GeneticAlgorithm, monkeypatch):
    ga = genetic_algorithm
    monkeypatch.setattr("random.random", lambda: 0.0)
    result = ga.run()
    assert isinstance(result["history"], list)
    assert all(c >= 0.0 for _, c in result["history"])


def test_seed_produces_deterministic_results(mock_problem, operator_factory):
    """Ensure the same seed produces identical optimization behavior (ignoring timing and iteration count)."""
    selection = operator_factory.get_operator("selection", "tournament", rate=0.5)
    crossover = operator_factory.get_operator("crossover", "ox")
    mutation = operator_factory.get_operator("mutation", "insert")
    succession = operator_factory.get_operator("succession", "elitist", elite_rate=0.5)

    def make_ga():
        return GeneticAlgorithm(
            problem=mock_problem,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            succession=succession,
            population_size=4,
            crossover_rate=0.9,
            mutation_rate=0.1,
            max_time=0.03,
            seed=42,
        )

    result1 = make_ga().run()
    result2 = make_ga().run()

    costs1 = [c for _, c in result1["history"]]
    costs2 = [c for _, c in result2["history"]]

    min_len = min(len(costs1), len(costs2))
    costs1 = costs1[:min_len]
    costs2 = costs2[:min_len]

    assert costs1 == costs2, (
        f"Cost trajectories differ despite identical seeds: "
        f"len1={len(result1['history'])}, len2={len(result2['history'])}"
    )
