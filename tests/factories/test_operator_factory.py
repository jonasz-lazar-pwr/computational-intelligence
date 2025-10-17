import pytest

from src.factories.operator_factory import OperatorFactory
from src.operators.crossover.cx import CycleCrossover
from src.operators.crossover.ox import OrderCrossover
from src.operators.crossover.pmx import PartiallyMappedCrossover
from src.operators.mutation.insert import InsertMutation
from src.operators.mutation.swap import SwapMutation
from src.operators.selection.rank import RankSelection
from src.operators.selection.roulette import RouletteSelection
from src.operators.selection.tournament import TournamentSelection
from src.operators.succession.elitist import ElitistSuccession
from src.operators.succession.steady_state import SteadyStateSuccession


def test_registry_structure_integrity():
    """Ensure OperatorFactory registry has expected categories and operator mappings."""
    reg = OperatorFactory._REGISTRY
    assert set(reg.keys()) == {"selection", "crossover", "mutation", "succession"}
    assert set(reg["selection"]) == {"tournament", "roulette", "rank"}
    assert set(reg["crossover"]) == {"ox", "cx", "pmx"}
    assert set(reg["mutation"]) == {"insert", "swap"}
    assert set(reg["succession"]) == {"elitist", "steady_state"}
    assert reg["selection"]["tournament"] is TournamentSelection
    assert reg["selection"]["roulette"] is RouletteSelection
    assert reg["mutation"]["swap"] is SwapMutation
    assert reg["succession"]["steady_state"] is SteadyStateSuccession


@pytest.mark.parametrize(
    "category,name,expected_class",
    [
        ("selection", "tournament", TournamentSelection),
        ("selection", "roulette", RouletteSelection),
        ("selection", "rank", RankSelection),
        ("crossover", "ox", OrderCrossover),
        ("crossover", "cx", CycleCrossover),
        ("crossover", "pmx", PartiallyMappedCrossover),
        ("mutation", "insert", InsertMutation),
        ("mutation", "swap", SwapMutation),
        ("succession", "elitist", ElitistSuccession),
        ("succession", "steady_state", SteadyStateSuccession),
    ],
)
def test_get_operator_returns_correct_instance(category, name, expected_class):
    """Check that get_operator returns correct operator type."""
    op = OperatorFactory.get_operator(category, name)
    assert isinstance(op, expected_class)


def test_get_operator_passes_kwargs(monkeypatch):
    """Ensure configuration kwargs are passed correctly to operator constructors."""
    captured = {}

    class DummyOp:
        def __init__(self, **cfg):
            captured.update(cfg)

    monkeypatch.setitem(OperatorFactory._REGISTRY["selection"], "dummy", DummyOp)
    op = OperatorFactory.get_operator("selection", "dummy", rate=0.5, custom=True)
    assert isinstance(op, DummyOp)
    assert captured == {"rate": 0.5, "custom": True}


@pytest.mark.parametrize(
    "category,name",
    [
        ("selection", "nonexistent"),
        ("crossover", "fake"),
        ("mutation", "none"),
        ("succession", "bad"),
        ("invalid_category", "whatever"),
    ],
)
def test_get_operator_raises_for_invalid_keys(category, name):
    """Invalid category or name should raise ValueError."""
    with pytest.raises(ValueError) as excinfo:
        OperatorFactory.get_operator(category, name)
    assert "Unknown operator" in str(excinfo.value)


def test_get_operator_creates_fresh_instance_each_call():
    """Ensure get_operator returns a new instance each time."""
    op1 = OperatorFactory.get_operator("mutation", "swap")
    op2 = OperatorFactory.get_operator("mutation", "swap")
    assert op1 is not op2
    assert isinstance(op1, SwapMutation)
    assert isinstance(op2, SwapMutation)
