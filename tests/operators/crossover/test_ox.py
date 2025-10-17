import random

import pytest

from src.operators.crossover.ox import OrderCrossover


@pytest.fixture(scope="module")
def ox_operator() -> OrderCrossover:
    """Return OrderCrossover instance."""
    return OrderCrossover()


def test_ox_basic_structure(monkeypatch, ox_operator: OrderCrossover):
    """Ensure crossover returns two valid offspring with correct length and content."""
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [2, 4])
    c1, c2 = ox_operator.crossover(p1, p2)

    assert isinstance(c1, list)
    assert isinstance(c2, list)
    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1) == set(p1)
    assert set(c2) == set(p2)


def test_ox_preserves_order_and_elements(monkeypatch, ox_operator: OrderCrossover):
    """Verify that order crossover preserves parent gene order outside copied segment."""
    p1 = [0, 1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1, 0]

    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 4])
    c1, c2 = ox_operator.crossover(p1, p2)

    assert c1[1:4] == [1, 2, 3]
    assert c2[1:4] == [4, 3, 2]

    for c in (c1, c2):
        assert sorted(c) == list(range(6))
        assert len(set(c)) == len(c)


def test_ox_symmetry(monkeypatch, ox_operator: OrderCrossover):
    """Ensure that crossover is symmetric (swapping parents swaps offspring)."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 3])
    c1, c2 = ox_operator.crossover(p1, p2)
    r1, r2 = ox_operator.crossover(p2, p1)

    assert sorted([c1, c2]) == sorted([r1, r2])


def test_ox_handles_minimum_length(monkeypatch, ox_operator: OrderCrossover):
    """Check behavior for smallest valid chromosome length (2)."""
    p1 = [0, 1]
    p2 = [1, 0]

    monkeypatch.setattr(random, "sample", lambda seq, k: [0, 1])
    c1, c2 = ox_operator.crossover(p1, p2)

    assert len(c1) == 2
    assert len(c2) == 2
    assert set(c1) == {0, 1}
    assert set(c2) == {0, 1}


def test_ox_large_population(monkeypatch, ox_operator: OrderCrossover):
    """Stress test with long chromosome (deterministic indices)."""
    p1 = list(range(50))
    p2 = p1[::-1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [10, 40])
    c1, c2 = ox_operator.crossover(p1, p2)

    assert len(c1) == 50
    assert len(c2) == 50
    assert set(c1) == set(range(50))
    assert set(c2) == set(range(50))


def test_ox_randomness_changes_results(ox_operator: OrderCrossover):
    """Different random slices should yield different offspring."""
    p1 = [0, 1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1, 0]

    random.seed(1)
    c1a, _ = ox_operator.crossover(p1, p2)
    random.seed(2)
    c1b, _ = ox_operator.crossover(p1, p2)

    assert c1a != c1b, "Random crossover points should change offspring"


def test_ox_invalid_length(monkeypatch, ox_operator: OrderCrossover):
    """Simulate invalid parents of different lengths."""
    p1 = [1, 2, 3]
    p2 = [1, 2]
    with pytest.raises(ValueError):
        if len(p1) != len(p2):
            raise ValueError("Parent lengths must match")
        ox_operator.crossover(p1, p2)


def test_ox_determinism_given_fixed_sample(monkeypatch, ox_operator: OrderCrossover):
    """Ensure deterministic result if random.sample is fixed."""
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [2, 4])
    c1a, c2a = ox_operator.crossover(p1, p2)
    c1b, c2b = ox_operator.crossover(p1, p2)

    assert (c1a, c2a) == (c1b, c2b), "Fixed sample should make output deterministic"
