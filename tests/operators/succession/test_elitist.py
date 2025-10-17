import pytest

from src.operators.succession.elitist import ElitistSuccession


@pytest.fixture
def sample_population():
    """Fixture providing dummy population with ascending costs."""
    parents = [[0], [1], [2], [3], [4]]
    parent_costs = [50, 40, 30, 20, 10]
    offspring = [[5], [6], [7], [8], [9]]
    offspring_costs = [45, 35, 25, 15, 5]
    return parents, parent_costs, offspring, offspring_costs


def test_basic_replacement_preserves_elites(sample_population):
    """Check that elites from parents are preserved and offspring fill the rest."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = ElitistSuccession(elite_rate=0.2)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert len(new_costs) == len(parents)
    assert [4] in new_pop
    assert min(new_costs) == 5
    assert max(new_costs) <= max(parent_costs)


def test_elite_rate_rounding_up_to_one(sample_population):
    """Elite count should be at least one, even for tiny elite_rate."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = ElitistSuccession(elite_rate=0.01)
    new_pop, _ = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert any(p in new_pop for p in parents)


def test_full_elite_rate_replaces_no_offspring(sample_population):
    """If elite_rate=1, all parents should be preserved (no offspring)."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = ElitistSuccession(elite_rate=1.0)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert set(tuple(p) for p in new_pop) == set(tuple(p) for p in parents)
    assert all(c in parent_costs for c in new_costs)


def test_invalid_elite_rate_raises():
    """Ensure ValueError for invalid elite_rate."""
    with pytest.raises(ValueError):
        ElitistSuccession(elite_rate=0)
    with pytest.raises(ValueError):
        ElitistSuccession(elite_rate=-0.5)
    with pytest.raises(ValueError):
        ElitistSuccession(elite_rate=1.5)


def test_preserves_order_and_best_cost(sample_population):
    """Ensure new population contains valid individuals and costs from both groups."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = ElitistSuccession(elite_rate=0.4)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert all(isinstance(p, list) for p in new_pop)

    all_costs = set(parent_costs + offspring_costs)

    assert all(c in all_costs for c in new_costs)
    assert min(new_costs) == min(parent_costs + offspring_costs)


def test_handles_minimal_population(monkeypatch):
    """Edge case: single parent and offspring."""
    parents = [[0]]
    offspring = [[1]]
    parent_costs = [10.0]
    offspring_costs = [5.0]
    op = ElitistSuccession(elite_rate=0.5)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert new_pop in ([[0]], [[1]])
    assert len(new_costs) == 1


def test_equal_costs_between_parents_and_offspring():
    """Ensure deterministic behavior when all costs are equal."""
    parents = [[0], [1], [2]]
    offspring = [[3], [4], [5]]
    parent_costs = [10.0, 10.0, 10.0]
    offspring_costs = [10.0, 10.0, 10.0]

    op = ElitistSuccession(elite_rate=0.33)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert all(isinstance(c, float) for c in new_costs)
    assert set(tuple(x) for x in new_pop) <= {tuple(x) for x in (parents + offspring)}
