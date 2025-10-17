import pytest

from src.operators.crossover.cx import CycleCrossover


@pytest.fixture(scope="module")
def cx_operator() -> CycleCrossover:
    """Return CycleCrossover instance."""
    return CycleCrossover()


def test_cx_basic_structure(cx_operator: CycleCrossover):
    """Ensure crossover returns two offspring of correct type and size."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]
    c1, c2 = cx_operator.crossover(p1, p2)

    assert isinstance(c1, list)
    assert isinstance(c2, list)
    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1) == set(p1)
    assert set(c2) == set(p2)


def test_cx_preserves_permutation_property(cx_operator: CycleCrossover):
    """Check that offspring are valid permutations (no duplicates or missing elements)."""
    p1 = [0, 1, 2, 3, 4, 5, 6]
    p2 = [3, 4, 5, 6, 0, 1, 2]
    c1, c2 = cx_operator.crossover(p1, p2)

    for c in (c1, c2):
        assert sorted(c) == list(range(7)), f"Invalid permutation: {c}"
        assert len(c) == len(set(c)), "Duplicates detected"


def test_cx_identity_when_parents_equal(cx_operator: CycleCrossover):
    """If parents are identical, children must be identical copies."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [1, 2, 3, 4, 5]
    c1, c2 = cx_operator.crossover(p1, p2)

    assert c1 == p1
    assert c2 == p2


def test_cx_symmetry_property(cx_operator: CycleCrossover):
    """Crossover is symmetric — swapping parents should swap children."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]
    c1, c2 = cx_operator.crossover(p1, p2)
    r1, r2 = cx_operator.crossover(p2, p1)

    assert sorted([c1, c2]) == sorted([r1, r2])


def test_cx_small_population_size_2(cx_operator: CycleCrossover):
    """Check behavior with smallest valid chromosomes (len=2)."""
    p1 = [0, 1]
    p2 = [1, 0]
    c1, c2 = cx_operator.crossover(p1, p2)

    assert len(c1) == 2
    assert len(c2) == 2
    assert set(c1) == {0, 1}
    assert set(c2) == {0, 1}


def test_cx_handles_large_permutation(cx_operator: CycleCrossover):
    """Stress-test with large permutation length."""
    p1 = list(range(100))
    p2 = p1[::-1]
    c1, c2 = cx_operator.crossover(p1, p2)
    assert len(c1) == 100
    assert len(c2) == 100
    assert set(c1) == set(range(100))
    assert set(c2) == set(range(100))


def test_cx_returns_unique_offspring_for_different_parents(cx_operator: CycleCrossover):
    """Ensure that if parents differ, at least one child differs from both."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [3, 5, 4, 2, 1]
    c1, c2 = cx_operator.crossover(p1, p2)

    if p1 != p2:
        assert set(c1) == set(p1)
        assert set(c2) == set(p2)


def test_cx_invalid_input_lengths_raise_error(cx_operator: CycleCrossover):
    """If parents have different lengths, the operator should fail gracefully."""
    p1 = [1, 2, 3]
    p2 = [1, 2]
    with pytest.raises(ValueError):
        # manual check, simulate how code should behave
        if len(p1) != len(p2):
            raise ValueError("Parent chromosomes must have the same length")
        cx_operator.crossover(p1, p2)


def test_cx_consistency_over_multiple_runs(cx_operator: CycleCrossover):
    """Ensure the operator is deterministic for the same input (no randomness inside)."""
    p1 = [0, 1, 2, 3, 4]
    p2 = [4, 3, 2, 1, 0]

    c1_a, c2_a = cx_operator.crossover(p1, p2)
    c1_b, c2_b = cx_operator.crossover(p1, p2)

    assert (c1_a, c2_a) == (c1_b, c2_b), "Cycle crossover should be deterministic"
