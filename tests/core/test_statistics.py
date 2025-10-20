import pytest

from src.core.statistics import Statistics


@pytest.fixture
def stats():
    return Statistics()


def test_compute_mean_error_empty_list(stats, caplog):
    """Should return 0.0 and log warning when no results provided."""
    with caplog.at_level("WARNING"):
        result = stats.compute_mean_error([], optimum=100.0)
    assert result == 0.0
    assert "No results provided" in caplog.text


def test_compute_mean_error_no_optimum(stats, caplog):
    """Should return mean cost when optimum is None."""
    with caplog.at_level("DEBUG"):
        result = stats.compute_mean_error([10, 20, 30], optimum=None)
    assert result == pytest.approx(20.0)
    assert "Optimum not provided" in caplog.text


def test_compute_mean_error_zero_optimum(stats, caplog):
    """Should handle zero optimum by returning mean cost."""
    with caplog.at_level("DEBUG"):
        result = stats.compute_mean_error([5, 15], optimum=0.0)
    assert result == pytest.approx(10.0)
    assert "Optimum not provided" in caplog.text


def test_compute_mean_error_valid_optimum(stats, caplog):
    """Should compute correct mean relative error."""
    with caplog.at_level("DEBUG"):
        result = stats.compute_mean_error([110, 90], optimum=100.0)
    # Relative errors: (10/100) + (-10/100) = 0 → mean = 0
    assert result == pytest.approx(0.0)
    assert "Computed mean relative error" in caplog.text


def test_best_cost_empty_list(stats, caplog):
    """Should return inf and log warning if no results."""
    with caplog.at_level("WARNING"):
        result = stats.best_cost([])
    assert result == float("inf")
    assert "No results provided" in caplog.text


def test_best_cost_valid_list(stats, caplog):
    """Should return minimum value from results."""
    with caplog.at_level("DEBUG"):
        result = stats.best_cost([12.5, 7.3, 9.1])
    assert result == pytest.approx(7.3)
    assert "Best cost found" in caplog.text
