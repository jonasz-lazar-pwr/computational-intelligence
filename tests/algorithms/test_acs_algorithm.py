import time

from typing import List

import pytest

from src.algorithms.acs_algorithm import ACSAlgorithm
from src.interfaces.problems_interfaces import IProblem


class DummyTSPProblem(IProblem):
    """Minimal TSP problem for ACS testing."""

    def __init__(self, dimension: int = 5) -> None:
        self._dimension = dimension
        self._dist = [[abs(i - j) + 1 for j in range(dimension)] for i in range(dimension)]

    def evaluate(self, route: List[int]) -> float:
        """Compute the cost of a given route."""
        total = 0
        for i in range(len(route)):
            total += self._dist[route[i]][route[(i + 1) % len(route)]]
        return float(total)

    def get_initial_solution(self) -> List[int]:
        """Return the base permutation."""
        return list(range(self._dimension))

    def get_dimension(self) -> int:
        """Return problem dimension."""
        return self._dimension

    def get_distance(self, i: int, j: int) -> float:
        """Return distance between cities."""
        return self._dist[i][j]

    def optimal_value(self):
        """Return no known optimum."""
        return None

    def info(self) -> dict:
        """Return problem metadata."""
        return {"name": "DummyTSP", "dimension": self._dimension}


@pytest.fixture()
def problem():
    """Provide DummyTSPProblem instance."""
    return DummyTSPProblem()


@pytest.fixture()
def make_acs(problem):
    """Return ACS factory."""

    def _factory(seed=1):
        return ACSAlgorithm(
            problem=problem,
            num_ants=5,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            phi=0.1,
            q0=0.9,
            max_time=0.05,
            seed=seed,
        )

    return _factory


def test_initialize_matrices(make_acs, problem):
    """Verify pheromone and heuristic matrices are initialized."""
    acs = make_acs()
    acs._initialize_pheromone_and_heuristic()

    n = problem.get_dimension()
    assert len(acs._pheromone) == n
    assert len(acs._heuristic) == n
    assert all(len(row) == n for row in acs._pheromone)
    assert all(len(row) == n for row in acs._heuristic)


def test_build_route_returns_valid_tour(make_acs, problem):
    """Ensure route contains each city exactly once."""
    acs = make_acs()
    acs._initialize_pheromone_and_heuristic()

    route = acs._build_route()

    assert len(route) == problem.get_dimension()
    assert set(route) == set(range(problem.get_dimension()))


def test_run_returns_history(make_acs):
    """Ensure run() returns history with time and cost pairs."""
    acs = make_acs()
    result = acs.run()

    assert "history" in result
    assert isinstance(result["history"], list)
    assert all(len(entry) == 2 for entry in result["history"])


def test_run_respects_time_limit(make_acs):
    """Ensure ACS respects time limit."""
    acs = make_acs()
    acs.max_time = 0.05

    start = time.time()
    acs.run()
    elapsed = time.time() - start

    assert elapsed < 0.5


def test_seed_produces_deterministic_results(make_acs):
    """Ensure identical seeds produce identical history."""
    acs1 = make_acs(seed=123)
    acs2 = make_acs(seed=123)

    h1 = [v for _, v in acs1.run()["history"]]
    h2 = [v for _, v in acs2.run()["history"]]

    min_len = min(len(h1), len(h2))
    assert h1[:min_len] == h2[:min_len]


def test_route_evaluates_without_error(make_acs, problem):
    """Ensure evaluate(route) works for generated route."""
    acs = make_acs()
    acs._initialize_pheromone_and_heuristic()

    route = acs._build_route()
    cost = problem.evaluate(route)

    assert cost > 0


def test_global_update_keeps_symmetry(make_acs, problem):
    """Ensure global pheromone update keeps symmetry."""
    acs = make_acs()
    acs._initialize_pheromone_and_heuristic()

    route = [0, 1, 2, 3, 4]
    cost = problem.evaluate(route)

    acs._global_update(route, cost)

    for i in range(problem.get_dimension()):
        for j in range(problem.get_dimension()):
            assert acs._pheromone[i][j] == acs._pheromone[j][i]


def test_local_update_decreases_pheromone(make_acs, problem):
    """Ensure local update decreases pheromone."""
    acs = make_acs()
    acs._initialize_pheromone_and_heuristic()

    before = acs._pheromone[0][1]
    acs._local_update(0, 1)
    after = acs._pheromone[0][1]

    assert after <= before + 1e-12
