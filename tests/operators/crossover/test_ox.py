import pytest

from src.operators.crossover.ox import OrderCrossover


@pytest.fixture()
def ox():
    """Provide OrderCrossover instance."""
    return OrderCrossover()


def test_crossover_preserves_elements(ox, monkeypatch):
    """Check offspring contain same elements and valid length."""
    p1 = [1, 2, 3, 4, 5]
    p2 = [5, 4, 3, 2, 1]
    monkeypatch.setattr("random.sample", lambda seq, k: [1, 3])
    c1, c2 = ox.crossover(p1, p2)
    assert sorted(c1) == sorted(p1)
    assert sorted(c2) == sorted(p2)
    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert all(isinstance(x, int) for x in c1 + c2)


def test_crossover_symmetry(ox, monkeypatch):
    """Check crossover produces distinct but valid children."""
    monkeypatch.setattr("random.sample", lambda seq, k: [0, 2])
    p1 = [1, 2, 3, 4]
    p2 = [4, 3, 2, 1]
    c1, c2 = ox.crossover(p1, p2)
    assert c1 != c2
    assert set(c1) == set(p1)


def test_crossover_randomness(monkeypatch):
    """Check random.sample receives correct arguments."""
    ox = OrderCrossover()
    called = {}

    def fake_sample(seq, k):
        called["args"] = (list(seq), k)
        return [0, 2]

    monkeypatch.setattr("random.sample", fake_sample)
    ox.crossover([1, 2, 3, 4], [4, 3, 2, 1])
    assert called["args"][1] == 2
    assert all(isinstance(x, int) for x in called["args"][0])


def test_crossover_edge_case_two_genes(ox, monkeypatch):
    """Check crossover handles minimal chromosome length."""
    monkeypatch.setattr("random.sample", lambda seq, k: [0, 1])
    p1 = [1, 2]
    p2 = [2, 1]
    c1, c2 = ox.crossover(p1, p2)
    assert c1 == p1 and c2 == p2
