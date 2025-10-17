import random

import pytest

from src.operators.selection.tournament import TournamentSelection


@pytest.fixture
def population_and_costs():
    """Fixture: small synthetic population and costs."""
    population = [
        [0, 1, 2],
        [2, 1, 0],
        [1, 0, 2],
        [2, 0, 1],
        [0, 2, 1],
    ]
    costs = [10.0, 5.0, 8.0, 3.0, 6.0]
    return population, costs


def test_tournament_init_valid():
    """Valid initialization with proper rate."""
    sel = TournamentSelection(rate=0.25)
    assert sel.rate == 0.25


@pytest.mark.parametrize("bad_rate", [0, -0.1, 1.5])
def test_tournament_init_invalid_rate_raises(bad_rate):
    """Invalid rate values should raise ValueError."""
    with pytest.raises(ValueError):
        TournamentSelection(rate=bad_rate)


def test_tournament_select_returns_valid_individual(monkeypatch, population_and_costs):
    """Selection should return one individual from population."""
    population, costs = population_and_costs

    monkeypatch.setattr(random, "sample", lambda seq, k: seq[:k])
    sel = TournamentSelection(rate=0.4)
    chosen = sel.select(population, costs)

    assert isinstance(chosen, list)
    assert chosen in population


def test_tournament_select_chooses_lowest_cost(monkeypatch, population_and_costs):
    """Ensure that the chosen individual has the lowest cost in the sample."""
    population, costs = population_and_costs

    def fake_sample(seq, k):
        return seq[:k]

    monkeypatch.setattr(random, "sample", fake_sample)
    sel = TournamentSelection(rate=0.6)
    result = sel.select(population, costs)

    assert result == [2, 1, 0]


def test_tournament_select_minimum_size(monkeypatch, population_and_costs):
    """Ensure minimum tournament size is 2."""
    population, costs = population_and_costs

    monkeypatch.setattr(random, "sample", lambda seq, k: seq[:k])
    sel = TournamentSelection(rate=0.01)
    result = sel.select(population, costs)

    assert result in population


def test_tournament_randomness_affects_result(population_and_costs):
    """Repeated runs may choose different individuals due to randomness."""
    population, costs = population_and_costs
    sel = TournamentSelection(rate=0.5)
    results = {tuple(sel.select(population, costs)) for _ in range(10)}

    assert len(results) > 1


def test_tournament_handles_small_population(monkeypatch):
    """Works even for very small populations."""
    population = [[0], [1]]
    costs = [2.0, 1.0]

    monkeypatch.setattr(random, "sample", lambda seq, k: seq[:k])
    sel = TournamentSelection(rate=1.0)
    result = sel.select(population, costs)

    assert result in population
    assert result == [1]
