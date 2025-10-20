import json

import pytest

from src.core.result_collector import ResultCollector
from src.interfaces.core_interfaces import IStatistics


class DummyStatistics(IStatistics):
    """Minimal mock for statistics interface."""

    def __init__(self):
        self._called = []

    def compute_mean_error(self, results, optimum):
        self._called.append(("mean", list(results), optimum))
        return sum(results) / len(results) if results else 0.0

    def best_cost(self, results):
        self._called.append(("best", list(results)))
        return min(results) if results else float("inf")


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path


@pytest.fixture
def dummy_stats():
    return DummyStatistics()


@pytest.fixture
def collector(tmp_output, dummy_stats):
    return ResultCollector(output_dir=tmp_output, statistics=dummy_stats)


def test_collect_run_creates_file_and_updates_cache(collector, tmp_output):
    """Should write JSON file and store best cost."""
    history = [(1.0, 10.0), (2.0, 8.0), (3.0, 7.5)]
    collector.collect_run("exp1", history)

    run_dir = tmp_output / "exp1"
    files = list(run_dir.glob("run_*.json"))
    assert len(files) == 1

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["config_name"] == "exp1"
    assert isinstance(data["history"], list)
    assert collector._results_cache["exp1"] == [7.5]


def test_collect_run_appends_multiple_runs(collector, tmp_output):
    """Should correctly increment run index and append to cache."""
    history = [(1.0, 12.0), (2.0, 11.0)]
    collector.collect_run("exp_multi", history)
    collector.collect_run("exp_multi", history)

    run_dir = tmp_output / "exp_multi"
    runs = list(run_dir.glob("run_*.json"))
    assert len(runs) == 2
    assert collector._results_cache["exp_multi"] == [11.0, 11.0]


def test_collect_run_skips_empty_history(collector, caplog):
    """Should skip run with empty history and log warning."""
    with caplog.at_level("WARNING"):
        collector.collect_run("exp_empty", [])
    assert "Skipping run save" in caplog.text
    assert "exp_empty" not in collector._results_cache


def test_finalize_config_creates_summary_and_clears_cache(collector, tmp_output):
    """Should compute stats, write summary.json, and clear cache."""
    config_name = "exp_summary"
    history = [(1.0, 10.0), (2.0, 8.0)]
    collector.collect_run(config_name, history)
    run_dir = tmp_output / config_name

    collector.finalize_config(config_name, optimal_value=5.0, runs=3)
    summary_path = run_dir / "summary.json"

    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["config_name"] == config_name
    assert summary["runs"] == 3
    assert "mean_error" in summary and "best_cost" in summary
    assert config_name not in collector._results_cache


def test_finalize_config_skips_when_no_results(collector, caplog):
    """Should log warning and not create file if no results exist."""
    with caplog.at_level("WARNING"):
        collector.finalize_config("exp_none", optimal_value=None, runs=2)
    assert "Summary skipped" in caplog.text
    assert not list((collector._base_path / "exp_none").glob("summary.json"))
