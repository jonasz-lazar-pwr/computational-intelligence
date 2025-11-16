import random

import pytest

from src.algorithms.acs_algorithm import ACSAlgorithm
from src.factories.algorithm_factory import AlgorithmFactory


class DummyProblem:
    """Minimal test problem for ACS."""

    def get_initial_solution(self):
        return [0, 1, 2]

    def evaluate(self, ind):
        return sum(ind)

    def get_dimension(self):
        return 3


@pytest.fixture
def base_acs_config():
    """Return minimal valid ACS configuration."""
    return {
        "problem": DummyProblem(),
        "num_ants": 5,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "phi": 0.1,
        "q0": 0.9,
        "max_time": 3.0,
        "seed": 123,
    }


def test_build_creates_acs_algorithm(base_acs_config):
    """Ensure ACSAlgorithm is constructed correctly."""
    algo = AlgorithmFactory.build("acs", **base_acs_config)

    assert isinstance(algo, ACSAlgorithm)
    assert algo.problem is base_acs_config["problem"]
    assert algo.num_ants == 5
    assert algo.alpha == 1.0
    assert algo.beta == 2.0
    assert algo.rho == 0.1
    assert algo.phi == 0.1
    assert algo.q0 == 0.9
    assert algo.max_time == 3.0

    assert hasattr(algo, "_rng")
    assert isinstance(algo._rng, random.Random)

    r1 = algo._rng.random()
    algo2 = AlgorithmFactory.build("acs", **base_acs_config)
    r2 = algo2._rng.random()
    assert r1 == r2


def test_build_acs_rejects_missing_fields(base_acs_config):
    """Raise KeyError when required ACS fields are missing."""
    required = ["num_ants", "alpha", "beta", "rho", "phi", "q0", "max_time"]

    for field in required:
        cfg = base_acs_config.copy()
        cfg.pop(field)

        with pytest.raises(KeyError):
            AlgorithmFactory.build("acs", **cfg)


def test_build_acs_ignores_ga_specific_fields(base_acs_config):
    """Ensure GA-only fields do not affect ACS building."""
    cfg = base_acs_config.copy()
    cfg.update(
        {
            "population_size": 500,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "selection_config": {"name": "tournament"},
            "mutation_config": {"name": "swap"},
        }
    )

    algo = AlgorithmFactory.build("acs", **cfg)
    assert isinstance(algo, ACSAlgorithm)


def test_build_acs_passes_problem_correctly(base_acs_config):
    """Ensure problem is passed correctly into ACSAlgorithm."""
    algo = AlgorithmFactory.build("acs", **base_acs_config)
    assert algo.problem is base_acs_config["problem"]


def test_build_acs_allows_missing_seed(base_acs_config):
    """Ensure seed parameter is optional."""
    cfg = base_acs_config.copy()
    cfg.pop("seed")

    algo = AlgorithmFactory.build("acs", **cfg)
    assert isinstance(algo, ACSAlgorithm)
    assert getattr(algo, "_seed", None) is None
