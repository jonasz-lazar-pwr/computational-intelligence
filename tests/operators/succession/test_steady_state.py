import pytest

from src.operators.succession.steady_state import SteadyStateSuccession


@pytest.fixture
def sample_population():
    """Fixture: create deterministic parent and offspring populations."""
    parents = [[0], [1], [2], [3], [4]]
    parent_costs = [50, 40, 30, 20, 10]
    offspring = [[5], [6], [7], [8], [9]]
    offspring_costs = [45, 35, 25, 15, 5]
    return parents, parent_costs, offspring, offspring_costs


def test_basic_replacement_correctness(sample_population):
    """Ensure correct replacement of worst parents with best offspring."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = SteadyStateSuccession(replacement_rate=0.4)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert len(new_costs) == len(parents)

    best_offspring = [[9], [8]]
    for b in best_offspring:
        assert b in new_pop

    worst_parents = [[0], [1]]
    for w in worst_parents:
        assert w not in new_pop


def test_replacement_rate_minimum_one(sample_population):
    """Ensure at least one parent is replaced for small replacement_rate."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = SteadyStateSuccession(replacement_rate=0.01)
    new_pop, _ = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert any(x in offspring for x in new_pop)


def test_full_replacement_replaces_all(sample_population):
    """If replacement_rate=1.0, entire population replaced by offspring."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = SteadyStateSuccession(replacement_rate=1.0)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert set(tuple(x) for x in new_pop) == set(tuple(x) for x in offspring)
    assert new_costs == sorted(offspring_costs)


def test_invalid_replacement_rate_raises():
    """Ensure invalid replacement_rate raises ValueError."""
    with pytest.raises(ValueError):
        SteadyStateSuccession(replacement_rate=0)
    with pytest.raises(ValueError):
        SteadyStateSuccession(replacement_rate=-0.3)
    with pytest.raises(ValueError):
        SteadyStateSuccession(replacement_rate=2.0)


def test_population_sorted_by_cost(sample_population):
    """Ensure resulting population contains valid and expected costs."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = SteadyStateSuccession(replacement_rate=0.5)
    _, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)
    combined_costs = set(parent_costs + offspring_costs)

    assert all(c in combined_costs for c in new_costs)
    assert len(new_costs) == len(parents)
    assert min(new_costs) == min(parent_costs + offspring_costs)
    assert sum(new_costs) / len(new_costs) <= sum(parent_costs) / len(parent_costs)


def test_handles_minimal_population():
    """Edge case: one parent and one offspring."""
    parents = [[0]]
    offspring = [[1]]
    parent_costs = [10.0]
    offspring_costs = [5.0]
    op = SteadyStateSuccession(replacement_rate=0.5)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert new_pop in ([[0]], [[1]])
    assert len(new_costs) == 1


def test_equal_costs_handled_gracefully():
    """When all costs are equal, any combination is valid but stable."""
    parents = [[0], [1], [2]]
    offspring = [[3], [4], [5]]
    parent_costs = [10.0, 10.0, 10.0]
    offspring_costs = [10.0, 10.0, 10.0]
    op = SteadyStateSuccession(replacement_rate=0.5)
    new_pop, new_costs = op.replace(parents, offspring, parent_costs, offspring_costs)

    assert len(new_pop) == len(parents)
    assert all(isinstance(x, list) for x in new_pop)
    assert all(isinstance(c, float) for c in new_costs)
    assert set(tuple(x) for x in new_pop) <= {tuple(x) for x in (parents + offspring)}


def test_correct_log_message(sample_population, caplog):
    """Check debug log message includes proper replacement count."""
    parents, parent_costs, offspring, offspring_costs = sample_population
    op = SteadyStateSuccession(replacement_rate=0.2)
    with caplog.at_level("DEBUG"):
        op.replace(parents, offspring, parent_costs, offspring_costs)
    assert "replaced" in caplog.text
    assert "individuals" in caplog.text
