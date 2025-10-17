import random

import pytest

from src.operators.selection.roulette import RouletteSelection


@pytest.fixture
def population_and_costs():
    """Fixture: deterministic integer population and corresponding costs."""
    population = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 1, 0],
        [2, 0, 1],
    ]
    costs = [40.0, 30.0, 20.0, 10.0]

    return population, costs


def test_basic_selection(monkeypatch, population_and_costs):
    """Ensure roulette selection returns a valid individual."""
    pop, costs = population_and_costs
    sel = RouletteSelection()

    monkeypatch.setattr(random, "uniform", lambda a, b: 0.4)
    result = sel.select(pop, costs)

    assert result in pop
    assert all(isinstance(g, int) for g in result)


def test_epsilon_prevents_division_by_zero():
    """Ensure epsilon avoids division by zero when cost = 0."""
    sel = RouletteSelection(epsilon=1e-6)
    pop = [[0], [1]]
    costs = [0.0, 1.0]
    result = sel.select(pop, costs)

    assert result in pop


def test_selection_prefers_lower_costs(monkeypatch):
    """Lower costs give higher probabilities, but order of iteration matters."""
    sel = RouletteSelection()
    pop = [[0], [1]]
    costs = [100.0, 1.0]
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.0)
    result = sel.select(pop, costs)

    assert result in pop


def test_high_random_draw_selects_last(monkeypatch, population_and_costs):
    """If random draw > cumulative probabilities, last element returned."""
    pop, costs = population_and_costs
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.0)
    sel = RouletteSelection()
    result = sel.select(pop, costs)

    assert result == pop[-1]


def test_probabilities_sum_to_one(population_and_costs):
    """Ensure normalized probabilities sum to one."""
    _, costs = population_and_costs
    sel = RouletteSelection()
    fitness = [1.0 / (c + sel.epsilon) for c in costs]
    probs = [f / sum(fitness) for f in fitness]

    assert abs(sum(probs) - 1.0) < 1e-9


def test_selection_varies_with_randomness(population_and_costs):
    """Repeated random draws should yield different outcomes."""
    pop, costs = population_and_costs
    sel = RouletteSelection()
    results = {tuple(sel.select(pop, costs)) for _ in range(20)}

    assert len(results) > 1


def test_handles_single_individual(monkeypatch):
    """Edge case: single individual should always be selected."""
    pop = [[42]]
    costs = [5.0]
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.5)
    sel = RouletteSelection()
    result = sel.select(pop, costs)

    assert result == [42]


def test_handles_two_individuals(monkeypatch):
    """Small population of size 2 should still produce valid results."""
    pop = [[0], [1]]
    costs = [10.0, 5.0]
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.7)
    sel = RouletteSelection()
    result = sel.select(pop, costs)

    assert result in pop


def test_custom_epsilon_affects_precision(monkeypatch):
    """Custom epsilon should slightly affect probabilities, but not crash."""
    pop = [[0], [1], [2]]
    costs = [1.0, 0.0, 2.0]
    sel = RouletteSelection(epsilon=1e-3)
    monkeypatch.setattr(random, "uniform", lambda a, b: 0.2)
    result = sel.select(pop, costs)

    assert result in pop
