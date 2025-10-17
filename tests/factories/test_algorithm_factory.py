from unittest.mock import MagicMock

import pytest

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.factories.algorithm_factory import AlgorithmFactory


class DummyProblem:
    """Minimal mock of IProblem for testing AlgorithmFactory."""

    def get_initial_solution(self):
        return [0, 1, 2]

    def evaluate(self, ind):
        return sum(ind)

    def get_dimension(self):
        return 3


@pytest.fixture
def base_config():
    """Return base configuration for algorithm construction."""
    return {
        "problem": DummyProblem(),
        "population_size": 10,
        "crossover_rate": 0.9,
        "mutation_rate": 0.1,
        "max_time": 1.0,
        "selection_config": {"name": "tournament", "rate": 0.5},
        "crossover_config": {"name": "ox"},
        "mutation_config": {"name": "swap"},
        "succession_config": {"name": "elitist", "rate": 0.2},
    }


def test_build_creates_genetic_algorithm(monkeypatch, base_config):
    """Verify factory builds a valid GeneticAlgorithm instance."""
    fake_op = MagicMock()

    monkeypatch.setattr(
        "src.factories.algorithm_factory.OperatorFactory.get_operator",
        lambda self, category, **cfg: fake_op,
    )

    algo = AlgorithmFactory.build("ga", **base_config)

    assert isinstance(algo, GeneticAlgorithm)
    assert algo.population_size == base_config["population_size"]
    assert algo.crossover_rate == 0.9
    assert algo.mutation_rate == 0.1
    assert algo.max_time == 1.0
    assert algo.selection is fake_op
    assert algo.crossover is fake_op
    assert algo.mutation is fake_op
    assert algo.succession is fake_op


def test_build_raises_for_unknown_algorithm():
    """Unknown algorithm name should raise ValueError."""
    with pytest.raises(ValueError) as e:
        AlgorithmFactory.build("nonexistent", problem=DummyProblem())
    assert "Unknown algorithm" in str(e.value)


def test_registry_contains_ga_entry():
    """Ensure GA is correctly registered in AlgorithmFactory."""
    assert "ga" in AlgorithmFactory._REGISTRY
    assert AlgorithmFactory._REGISTRY["ga"] is GeneticAlgorithm


def test_build_invokes_operator_factory(monkeypatch, base_config):
    """Ensure all operator types are requested from OperatorFactory."""
    called = []

    def fake_get_operator(self, category, **cfg):
        called.append(category)
        return MagicMock()

    monkeypatch.setattr(
        "src.factories.algorithm_factory.OperatorFactory.get_operator", fake_get_operator
    )

    AlgorithmFactory.build("ga", **base_config)
    assert set(called) == {"selection", "crossover", "mutation", "succession"}


def test_build_passes_correct_configs(monkeypatch, base_config):
    """Ensure configuration dicts are forwarded correctly."""
    captured = {}

    def fake_get_operator(self, category, **cfg):
        captured[category] = cfg
        return MagicMock()

    monkeypatch.setattr(
        "src.factories.algorithm_factory.OperatorFactory.get_operator", fake_get_operator
    )

    AlgorithmFactory.build("ga", **base_config)
    assert captured["selection"] == base_config["selection_config"]
    assert captured["crossover"] == base_config["crossover_config"]
    assert captured["mutation"] == base_config["mutation_config"]
    assert captured["succession"] == base_config["succession_config"]


def test_build_returns_distinct_operator_instances(monkeypatch, base_config):
    """Each operator type should produce a distinct instance."""

    def fake_get_operator(self, category, **cfg):
        return MagicMock(name=f"Fake_{category}")

    monkeypatch.setattr(
        "src.factories.algorithm_factory.OperatorFactory.get_operator", fake_get_operator
    )

    algo = AlgorithmFactory.build("ga", **base_config)
    ops = [algo.selection, algo.crossover, algo.mutation, algo.succession]
    assert len(set(id(x) for x in ops)) == 4


def test_build_propagates_problem_object(monkeypatch, base_config):
    """Ensure the problem instance is correctly passed to algorithm."""
    fake_op = MagicMock()

    monkeypatch.setattr(
        "src.factories.algorithm_factory.OperatorFactory.get_operator",
        lambda self, category, **cfg: fake_op,
    )

    algo = AlgorithmFactory.build("ga", **base_config)
    assert algo.problem is base_config["problem"]
