import random

import pytest

from src.operators.mutation.swap import SwapMutation


@pytest.fixture
def swap_operator() -> SwapMutation:
    return SwapMutation()


def test_swap_does_nothing_for_small_individuals(swap_operator: SwapMutation):
    """Mutation should not modify individuals shorter than minimum length."""
    for ind in ([], [1]):
        original = ind.copy()
        swap_operator.mutate(ind)
        assert ind == original, f"Individual {ind} should remain unchanged"


def test_swap_performs_valid_swap(monkeypatch, swap_operator: SwapMutation):
    """Mutation should correctly swap two positions."""
    individual = [0, 1, 2, 3, 4]
    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 3])
    swap_operator.mutate(individual)

    assert individual == [0, 3, 2, 1, 4]


def test_swap_random_indices_are_within_bounds(monkeypatch, swap_operator: SwapMutation):
    """Ensure selected indices are always valid for any population length."""
    called_indices = []

    orig_sample = random.sample

    def fake_sample(seq, k):
        res = orig_sample(seq, k)
        called_indices.append(tuple(res))
        return res

    monkeypatch.setattr(random, "sample", fake_sample)

    individual = list(range(10))
    swap_operator.mutate(individual)

    for i, j in called_indices:
        assert 0 <= i < len(individual)
        assert 0 <= j < len(individual)
        assert i != j


def test_swap_is_in_place(swap_operator: SwapMutation):
    """Ensure mutation modifies individual in-place, not returning new object."""
    individual = [1, 2, 3, 4]
    id_before = id(individual)
    swap_operator.mutate(individual)

    assert id(individual) == id_before


def test_swap_preserves_genes(monkeypatch, swap_operator: SwapMutation):
    """Mutation must not lose or duplicate genes."""
    individual = [1, 2, 3, 4, 5]
    monkeypatch.setattr(random, "sample", lambda seq, k: [0, 4])
    original_set = set(individual)
    swap_operator.mutate(individual)

    assert set(individual) == original_set
    assert sorted(individual) == sorted(original_set)


def test_swap_with_repeated_runs_produces_different_results(swap_operator: SwapMutation):
    """Repeated mutation calls should sometimes alter the order."""
    individual = [0, 1, 2, 3, 4]
    different_found = False

    for _ in range(20):
        temp = individual.copy()
        swap_operator.mutate(temp)
        if temp != individual:
            different_found = True
            break

    assert different_found, "Mutation should sometimes change order of elements"
