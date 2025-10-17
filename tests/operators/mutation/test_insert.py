import random

import pytest

from src.operators.mutation.insert import InsertMutation


@pytest.fixture
def insert_operator() -> InsertMutation:
    """Fixture for InsertMutation instance."""
    return InsertMutation()


def test_insert_performs_valid_move(monkeypatch, insert_operator: InsertMutation):
    """Check that one element is correctly moved to another position."""
    individual = [0, 1, 2, 3, 4]
    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 3])
    insert_operator.mutate(individual)

    assert individual == [0, 2, 3, 1, 4]


def test_insert_preserves_all_genes(monkeypatch, insert_operator: InsertMutation):
    """Ensure mutation keeps all original genes without duplicates or losses."""
    individual = [10, 20, 30, 40, 50]
    monkeypatch.setattr(random, "sample", lambda seq, k: [4, 0])
    insert_operator.mutate(individual)

    assert set(individual) == {10, 20, 30, 40, 50}
    assert len(individual) == 5


def test_insert_is_in_place(insert_operator: InsertMutation):
    """Mutation modifies individual in-place."""
    individual = [1, 2, 3, 4]
    id_before = id(individual)
    insert_operator.mutate(individual)

    assert id(individual) == id_before


def test_insert_with_repeated_runs_produces_variety(insert_operator: InsertMutation):
    """Repeated mutation should sometimes change order."""
    base = [0, 1, 2, 3, 4]
    changed = False

    for _ in range(20):
        temp = base.copy()
        insert_operator.mutate(temp)
        if temp != base:
            changed = True
            break

    assert changed, "Mutation should sometimes alter order of elements"


def test_insert_handles_two_elements(monkeypatch, insert_operator: InsertMutation):
    """Check smallest valid chromosome length of 2."""
    individual = [0, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [0, 1])
    insert_operator.mutate(individual)

    assert set(individual) == {0, 1}
    assert len(individual) == 2


def test_insert_random_indices_are_valid(monkeypatch, insert_operator: InsertMutation):
    """Ensure indices chosen by random.sample are valid within bounds."""
    captured = []
    orig_sample = random.sample

    def fake_sample(seq, k):
        res = orig_sample(seq, k)
        captured.append(tuple(res))
        return res

    monkeypatch.setattr(random, "sample", fake_sample)
    ind = list(range(10))
    insert_operator.mutate(ind)

    for i, j in captured:
        assert 0 <= i < len(ind)
        assert 0 <= j < len(ind)
        assert i != j
