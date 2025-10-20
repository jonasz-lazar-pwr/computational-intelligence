from unittest.mock import MagicMock, patch

import pytest

from src.core.experiment_runner import ExperimentRunner
from src.core.models import ExperimentConfig


@pytest.fixture
def mock_collector():
    """Provide mock IResultCollector."""
    collector = MagicMock()
    collector.collect_run = MagicMock()
    collector.finalize_config = MagicMock()
    return collector


@pytest.fixture
def sample_config():
    """Provide minimal ExperimentConfig instance."""
    return ExperimentConfig(
        name="exp_test",
        runs=2,
        seed_base=100,
        problem={"name": "tsp", "file_path": "data/ulysses16.tsp"},
        algorithm={
            "name": "ga",
            "population_size": 10,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "max_time": 0.01,
            "selection_config": {"name": "tournament"},
            "crossover_config": {"name": "ox"},
            "mutation_config": {"name": "insert"},
            "succession_config": {"name": "elitist"},
        },
    )


def test_run_all_calls_single_for_each_config(mock_collector, sample_config):
    """Run all configs exactly once per entry."""
    runner = ExperimentRunner(mock_collector)
    with patch.object(runner, "_run_single", autospec=True) as mock_run:
        runner.run_all([sample_config, sample_config])
        assert mock_run.call_count == 2


def test_run_all_logs_warning_for_empty_list(mock_collector, caplog):
    """Log warning if config list is empty."""
    runner = ExperimentRunner(mock_collector)
    with caplog.at_level("WARNING"):
        runner.run_all([])
    assert "No experiment configurations to run" in caplog.text


@patch("src.core.experiment_runner.ProblemFactory")
@patch("src.core.experiment_runner.AlgorithmFactory")
def test_single_run_executes_collects_and_finalizes(
    mock_algo_factory, mock_problem_factory, mock_collector, sample_config
):
    """Build problem, run algorithm, and finalize results."""
    mock_problem = MagicMock()
    mock_problem.optimal_value.return_value = 123.0
    mock_problem_factory.build.return_value = mock_problem

    mock_algorithm = MagicMock()
    mock_algorithm.run.return_value = {"history": [(1.0, 5.0), (2.0, 4.0)]}
    mock_algo_factory.build.return_value = mock_algorithm

    runner = ExperimentRunner(mock_collector)
    runner._run_single(sample_config)

    assert mock_problem_factory.build.called
    assert mock_algo_factory.build.call_count == sample_config.runs

    seeds_used = [
        call.kwargs["seed"] for call in mock_algo_factory.build.mock_calls if "seed" in call.kwargs
    ]
    assert seeds_used == [sample_config.seed_base + i for i in range(1, sample_config.runs + 1)]

    assert mock_collector.collect_run.call_count == sample_config.runs
    mock_collector.finalize_config.assert_called_once_with(
        sample_config.name, 123.0, sample_config.runs
    )


@patch("src.core.experiment_runner.ProblemFactory")
@patch("src.core.experiment_runner.AlgorithmFactory")
def test_single_run_handles_missing_optimal_value(
    mock_algo_factory, mock_problem_factory, mock_collector, sample_config
):
    """Handle problems without optimal_value method."""
    mock_problem = MagicMock()
    delattr(mock_problem, "optimal_value")
    mock_problem_factory.build.return_value = mock_problem
    mock_algo_factory.build.return_value.run.return_value = {"history": [(0.5, 9.0)]}

    runner = ExperimentRunner(mock_collector)
    runner._run_single(sample_config)

    mock_collector.finalize_config.assert_called_once_with(
        sample_config.name, None, sample_config.runs
    )


@patch("src.core.experiment_runner.ProblemFactory")
@patch("src.core.experiment_runner.AlgorithmFactory")
def test_run_all_catches_exceptions_and_logs(_, mock_collector, sample_config, caplog):
    """Catch exceptions and log errors."""
    runner = ExperimentRunner(mock_collector)
    with patch.object(runner, "_run_single", side_effect=ValueError("boom")):
        with caplog.at_level("ERROR"):
            runner.run_all([sample_config])
    assert "Error during execution" in caplog.text
    assert "boom" in caplog.text
