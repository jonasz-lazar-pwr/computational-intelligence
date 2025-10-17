from unittest.mock import MagicMock, patch

import pytest

from src.factories.problem_factory import ProblemFactory
from src.problems.tsp.tsp_instance import TSPInstance
from src.problems.tsp.tsp_problem import TSPProblem


def test_registry_contains_tsp_pair():
    """Ensure registry correctly maps 'tsp' to (TSPInstance, TSPProblem)."""
    assert "tsp" in ProblemFactory._REGISTRY
    instance_cls, problem_cls = ProblemFactory._REGISTRY["tsp"]
    assert instance_cls is TSPInstance
    assert problem_cls is TSPProblem


def test_build_returns_problem_instance(monkeypatch):
    """Check that ProblemFactory correctly builds TSPProblem with TSPInstance."""
    fake_instance = MagicMock()
    fake_problem = MagicMock()

    monkeypatch.setitem(
        ProblemFactory._REGISTRY, "tsp", (lambda **cfg: fake_instance, lambda inst: fake_problem)
    )

    result = ProblemFactory.build("tsp", file_path="a.tsp", optimal_results_path="b.json")
    assert result is fake_problem


def test_build_passes_correct_config(monkeypatch):
    """Ensure config dict is passed directly to TSPInstance constructor."""
    captured_cfg = {}

    class DummyInstance:
        def __init__(self, **cfg):
            captured_cfg.update(cfg)

    class DummyProblem:
        def __init__(self, instance):
            self.instance = instance

    monkeypatch.setitem(ProblemFactory._REGISTRY, "tsp", (DummyInstance, DummyProblem))

    ProblemFactory.build("tsp", file_path="input.tsp", optimal_results_path="opt.json")
    assert captured_cfg == {"file_path": "input.tsp", "optimal_results_path": "opt.json"}


def test_build_returns_tsp_problem_type(monkeypatch):
    """Ensure the default 'tsp' entry returns a TSPProblem instance."""
    with (
        patch.object(TSPInstance, "__init__", lambda self, **cfg: None),
        patch.object(TSPProblem, "__init__", lambda self, instance: None),
    ):
        result = ProblemFactory.build("tsp", file_path="x.tsp", optimal_results_path="y.json")
        assert isinstance(result, TSPProblem)


def test_build_raises_for_unknown_problem():
    """Unknown problem name should raise ValueError."""
    with pytest.raises(ValueError) as e:
        ProblemFactory.build("nonexistent", file_path="x.tsp")
    assert "Unknown problem" in str(e.value)


def test_build_creates_new_instances_each_call(monkeypatch):
    """Ensure separate calls produce new instance objects."""

    class DummyInstance:
        pass

    class DummyProblem:
        def __init__(self, instance):
            self.instance = instance

    monkeypatch.setitem(ProblemFactory._REGISTRY, "tsp", (DummyInstance, DummyProblem))

    p1 = ProblemFactory.build("tsp")
    p2 = ProblemFactory.build("tsp")
    assert p1 is not p2
    assert p1.instance is not p2.instance
