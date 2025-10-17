import random

import pytest

from src.operators.selection.rank import RankSelection


@pytest.fixture
def population_and_costs():
    """Fixture: deterministic integer population with associated costs."""
    population = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [2, 0, 1],
    ]
    costs = [40.0, 30.0, 20.0, 10.0]
    return population, costs


def test_rank_select_basic(monkeypatch, population_and_costs):
    """Check selection returns a valid individual."""
    pop, costs = population_and_costs
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.3)
    sel = RankSelection()
    result = sel.select(pop, costs)

    assert result in pop
    assert all(isinstance(gene, int) for gene in result)


def test_rank_select_best_for_low_r(monkeypatch, population_and_costs):
    """If random draw is very small, should select the best individual."""
    pop, costs = population_and_costs
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    sel = RankSelection()
    result = sel.select(pop, costs)

    assert result == [2, 0, 1]


def test_rank_select_worst_for_high_r(monkeypatch, population_and_costs):
    """If random draw exceeds all cumulative probabilities, select worst."""
    pop, costs = population_and_costs
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)
    sel = RankSelection()
    result = sel.select(pop, costs)

    assert result == [0, 1, 2]


def test_rank_selection_probability_distribution(population_and_costs):
    """Validate that rank-based probabilities sum to 1."""
    pop, costs = population_and_costs
    ranked = sorted(zip(pop, costs, strict=False), key=lambda x: x[1])
    n = len(ranked)
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [r / total for r in ranks]

    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p > 0 for p in probs)


def test_rank_selection_randomness_varies(population_and_costs):
    """Repeated runs with different random draws should produce variation."""
    pop, costs = population_and_costs
    sel = RankSelection()
    results = {tuple(sel.select(pop, costs)) for _ in range(20)}

    assert len(results) > 1


def test_rank_selection_handles_single_individual(monkeypatch):
    """Edge case: population of size 1 should always return that individual."""
    pop = [[42]]
    costs = [1.0]
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.5)
    sel = RankSelection()
    result = sel.select(pop, costs)

    assert result == [42]


def test_rank_selection_handles_two_individuals(monkeypatch):
    """Ensure correct behavior for minimal nontrivial case (n=2)."""
    pop = [[0], [1]]
    costs = [2.0, 1.0]
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.75)
    sel = RankSelection()
    result = sel.select(pop, costs)

    assert result in pop
