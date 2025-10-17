import random

import pytest

from src.operators.crossover.pmx import PartiallyMappedCrossover


@pytest.fixture(scope="module")
def pmx_operator() -> PartiallyMappedCrossover:
    """Return PartiallyMappedCrossover instance."""
    return PartiallyMappedCrossover()


def test_pmx_basic_structure(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Ensure crossover returns two valid offspring of correct type and content."""
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 4])
    c1, c2 = pmx_operator.crossover(p1, p2)

    assert isinstance(c1, list)
    assert isinstance(c2, list)
    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1) == set(p1)
    assert set(c2) == set(p2)


def test_pmx_mapping_correctness(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Verify that mapping is correctly applied to avoid duplicates."""
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [2, 5])
    c1, c2 = pmx_operator.crossover(p1, p2)

    assert c1[2:5] == [3, 4, 5]
    assert c2[2:5] == [4, 3, 2]

    assert len(set(c1)) == len(c1)
    assert len(set(c2)) == len(c2)
    assert sorted(c1) == sorted(p1)
    assert sorted(c2) == sorted(p2)


def test_pmx_symmetry(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Check that PMX crossover is symmetric."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 3])
    c1, c2 = pmx_operator.crossover(p1, p2)
    r1, r2 = pmx_operator.crossover(p2, p1)

    assert sorted([c1, c2]) == sorted([r1, r2])


def test_pmx_handles_small_length(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Check behavior for smallest valid chromosome length (2)."""
    p1 = [0, 1]
    p2 = [1, 0]

    monkeypatch.setattr(random, "sample", lambda seq, k: [0, 1])
    c1, c2 = pmx_operator.crossover(p1, p2)

    assert len(c1) == 2
    assert len(c2) == 2
    assert set(c1) == {0, 1}
    assert set(c2) == {0, 1}


def test_pmx_handles_large_population(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Stress test for large chromosomes with deterministic indices."""
    p1 = list(range(100))
    p2 = p1[::-1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [10, 40])
    c1, c2 = pmx_operator.crossover(p1, p2)

    assert len(c1) == 100
    assert len(c2) == 100
    assert set(c1) == set(range(100))
    assert set(c2) == set(range(100))


def test_pmx_randomness_changes_results(pmx_operator: PartiallyMappedCrossover):
    """Different random crossover points should produce different offspring."""
    p1 = [1, 2, 3, 4, 5, 6]
    p2 = [6, 5, 4, 3, 2, 1]

    random.seed(1)
    c1a, _ = pmx_operator.crossover(p1, p2)
    random.seed(2)
    c1b, _ = pmx_operator.crossover(p1, p2)

    assert c1a != c1b, "Different random slices should lead to different offspring"


def test_pmx_invalid_length(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Simulate invalid parents with different lengths."""
    p1 = [1, 2, 3]
    p2 = [1, 2]
    with pytest.raises(ValueError):
        if len(p1) != len(p2):
            raise ValueError("Parent lengths must match")
        pmx_operator.crossover(p1, p2)


def test_pmx_determinism_given_fixed_sample(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Ensure deterministic output if random.sample is fixed."""
    p1 = [0, 1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1, 0]

    monkeypatch.setattr(random, "sample", lambda seq, k: [2, 5])
    c1a, c2a = pmx_operator.crossover(p1, p2)
    c1b, c2b = pmx_operator.crossover(p1, p2)

    assert (c1a, c2a) == (c1b, c2b), "Fixed sample should ensure deterministic result"


def test_pmx_handles_all_elements_mapped(monkeypatch, pmx_operator: PartiallyMappedCrossover):
    """Edge case: mapping chain fully propagates through several elements."""
    p1 = [1, 2, 3, 4]
    p2 = [2, 3, 4, 1]

    monkeypatch.setattr(random, "sample", lambda seq, k: [1, 3])
    c1, c2 = pmx_operator.crossover(p1, p2)

    for c in (c1, c2):
        assert sorted(c) == [1, 2, 3, 4]
        assert len(set(c)) == 4
