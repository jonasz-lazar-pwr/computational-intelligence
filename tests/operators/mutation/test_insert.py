import pytest

from src.operators.mutation.insert import InsertMutation


@pytest.fixture()
def mutation():
    """Provide InsertMutation instance."""
    return InsertMutation()


def test_mutate_changes_order(monkeypatch, mutation):
    """Check mutation moves one element to new position."""
    individual = [1, 2, 3, 4]
    monkeypatch.setattr("random.sample", lambda seq, k: [0, 2])
    mutation.mutate(individual)
    assert individual != [1, 2, 3, 4]
    assert sorted(individual) == [1, 2, 3, 4]


def test_mutate_stability(mutation):
    """Check mutation preserves all elements and length."""
    individual = [1, 2, 3, 4, 5]
    mutation.mutate(individual)
    assert len(individual) == 5
    assert set(individual) == {1, 2, 3, 4, 5}


def test_mutate_with_minimal_individual(monkeypatch, mutation):
    """Check mutation works for minimal chromosome length."""
    monkeypatch.setattr("random.sample", lambda seq, k: [0, 1])
    individual = [1, 2]
    mutation.mutate(individual)
    assert sorted(individual) == [1, 2]
    assert len(individual) == 2


def test_mutate_random_sample_called(monkeypatch):
    """Check random.sample called with expected parameters."""
    mutation = InsertMutation()
    called = {}

    def fake_sample(seq, k):
        called["args"] = (list(seq), k)
        return [0, 1]

    monkeypatch.setattr("random.sample", fake_sample)
    mutation.mutate([1, 2, 3])
    assert called["args"][1] == 2
