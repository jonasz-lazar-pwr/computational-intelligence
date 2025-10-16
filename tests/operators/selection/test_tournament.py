import pytest

from src.operators.selection.tournament import TournamentSelection


@pytest.fixture()
def selection():
    """Provide TournamentSelection instance."""
    return TournamentSelection(tournament_size=2)


def test_select_returns_best_from_tournament(monkeypatch, selection):
    """Check best (lowest-cost) individual wins tournament."""
    population = [[1], [2], [3]]
    costs = [10, 5, 20]
    monkeypatch.setattr(
        "random.sample", lambda seq, k: list(zip(population, costs, strict=False))[:k]
    )
    result = selection.select(population, costs)
    assert result == [2]


def test_select_randomness_size(monkeypatch):
    """Check random.sample called with correct parameters."""
    selection = TournamentSelection(tournament_size=3)
    called = {}

    def fake_sample(seq, k):
        called["args"] = (seq, k)
        return [seq[0]]

    monkeypatch.setattr("random.sample", fake_sample)
    population, costs = [[1], [2], [3]], [10, 20, 30]
    selection.select(population, costs)
    seq, k = called["args"]
    assert isinstance(seq, list)
    assert k == 3


def test_select_with_single_participant(monkeypatch):
    """Check selection works with single tournament participant."""
    selection = TournamentSelection(tournament_size=1)
    population = [[1], [2], [3]]
    costs = [10, 20, 30]
    monkeypatch.setattr("random.sample", lambda seq, k: [(population[1], costs[1])])
    result = selection.select(population, costs)
    assert result == [2]


def test_select_handles_equal_costs(monkeypatch):
    """Check selection behavior when costs are equal."""
    selection = TournamentSelection(tournament_size=2)
    population = [[1], [2]]
    costs = [10, 10]
    monkeypatch.setattr(
        "random.sample", lambda seq, k: list(zip(population, costs, strict=False))[:k]
    )
    result = selection.select(population, costs)
    assert result in population
