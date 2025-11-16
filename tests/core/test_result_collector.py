import json

import pytest

from src.core.result_collector import ResultCollector
from src.interfaces.core_interfaces import IStatistics


class DummyStatistics(IStatistics):
    """Minimal mock implementing statistics methods."""

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
    """Return temporary output directory."""
    return tmp_path


@pytest.fixture
def dummy_stats():
    """Return dummy statistics instance."""
    return DummyStatistics()


@pytest.fixture
def collector(tmp_output, dummy_stats):
    """Return ResultCollector instance."""
    return ResultCollector(output_dir=tmp_output, statistics=dummy_stats)


def test_collect_run_adds_to_cache(collector):
    """Cache best cost extracted from history."""
    history = [(1.0, 10.0), (2.0, 8.0), (3.0, 7.5)]
    collector.collect_run("exp1", history)
    assert "exp1" in collector._results_cache
    assert collector._results_cache["exp1"] == [7.5]


def test_collect_run_multiple_accumulates_runs(collector):
    """Accumulate best costs for multiple runs."""
    history = [(1.0, 12.0), (2.0, 11.0)]
    collector.collect_run("exp_multi", history)
    collector.collect_run("exp_multi", history)
    assert collector._results_cache["exp_multi"] == [11.0, 11.0]


def test_collect_run_skips_empty_history(collector, caplog):
    """Skip run when history is empty."""
    with caplog.at_level("WARNING"):
        collector.collect_run("exp_empty", [])
    assert "Skipping run collection" in caplog.text
    assert "exp_empty" not in collector._results_cache


def test_finalize_config_appends_to_results(collector, tmp_output):
    """Append statistics for one config into results.json."""
    config_name = "exp_results"
    history = [(1.0, 10.0), (2.0, 8.0)]
    collector.collect_run(config_name, history)

    collector.finalize_config(config_name, optimal_value=5.0, runs=3)
    results_path = tmp_output / "results.json"

    assert results_path.exists()
    content = json.loads(results_path.read_text(encoding="utf-8"))
    assert isinstance(content, list)
    assert len(content) == 1

    entry = content[0]
    assert entry["config_name"] == config_name
    assert entry["runs"] == 3
    assert "mean_error" in entry and "best_cost" in entry
    assert config_name not in collector._results_cache


def test_finalize_config_appends_multiple_entries(collector, tmp_output):
    """Append multiple entries sequentially."""
    history = [(1.0, 9.0), (2.0, 8.5)]
    collector.collect_run("exp_A", history)
    collector.finalize_config("exp_A", optimal_value=5.0, runs=1)

    collector.collect_run("exp_B", [(1.0, 15.0), (2.0, 14.0)])
    collector.finalize_config("exp_B", optimal_value=10.0, runs=2)

    results_path = tmp_output / "results.json"
    content = json.loads(results_path.read_text(encoding="utf-8"))

    assert len(content) == 2
    names = [c["config_name"] for c in content]
    assert {"exp_A", "exp_B"} <= set(names)


def test_finalize_config_handles_invalid_results_json(collector, tmp_output):
    """Recreate results.json when corrupted."""
    results_path = tmp_output / "results.json"
    results_path.write_text("{invalid json", encoding="utf-8")

    collector._results_cache["exp_corrupt"] = [10.0, 12.0]
    collector.finalize_config("exp_corrupt", optimal_value=None, runs=1)

    content = json.loads(results_path.read_text(encoding="utf-8"))
    assert len(content) == 1
    assert content[0]["config_name"] == "exp_corrupt"


def test_finalize_config_skips_when_no_results(collector, caplog, tmp_output):
    """Skip finalize when no cached results exist."""
    results_path = tmp_output / "results.json"
    results_path.write_text(json.dumps([], indent=2), encoding="utf-8")

    with caplog.at_level("WARNING"):
        collector.finalize_config("exp_none", optimal_value=None, runs=2)

    assert "Skipping results append" in caplog.text
    content = json.loads(results_path.read_text(encoding="utf-8"))
    assert content == []
