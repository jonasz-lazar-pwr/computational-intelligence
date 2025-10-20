import pytest

from src.core.config_expander import ConfigExpander
from src.core.models import ExperimentConfig
from src.interfaces.core_interfaces import IConfigValidator, INameGenerator


class DummyValidator(IConfigValidator):
    """Minimal validator mock for configuration testing."""

    def __init__(self):
        self.validated_problems = []
        self.validated_algorithms = []

    def validate_root(self, data):
        pass

    def validate_problem(self, problem):
        self.validated_problems.append(problem)

    def validate_algorithm(self, algorithm, allow_lists):
        self.validated_algorithms.append((algorithm, allow_lists))


class DummyNamer(INameGenerator):
    """Minimal name generator mock."""

    def __init__(self):
        self.calls = []

    def generate(self, problem, alg, prefix):
        name = f"{prefix}_{alg.get('population_size', 'X')}"
        self.calls.append((problem, alg, prefix))
        return name


@pytest.fixture
def validator():
    return DummyValidator()


@pytest.fixture
def namer():
    return DummyNamer()


@pytest.fixture
def expander(validator, namer):
    return ConfigExpander(validator, namer)


def test_expand_manual_creates_experiment_configs(expander, validator):
    """Create single ExperimentConfig from manual definition."""
    data = {
        "experiments": [
            {
                "name": "exp1",
                "runs": 10,
                "seed_base": 42,
                "problem": {"file_path": "a.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {"name": "ga", "population_size": 100},
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 1
    cfg = result[0]
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "exp1"
    assert cfg.runs == 10
    assert cfg.seed_base == 42
    assert validator.validated_problems == [cfg.problem]
    assert validator.validated_algorithms[0][1] is False


def test_expand_manual_handles_multiple_configs(expander):
    """Handle multiple manual experiment definitions."""
    data = {
        "experiments": [
            {
                "name": "expA",
                "runs": 3,
                "seed_base": 1,
                "problem": {"file_path": "x.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {"name": "ga", "population_size": 100},
            },
            {
                "name": "expB",
                "runs": 5,
                "seed_base": 2,
                "problem": {"file_path": "y.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {"name": "ga", "population_size": 200},
            },
        ]
    }
    result = expander.expand(data)
    assert [c.name for c in result] == ["expA", "expB"]


def test_expand_sweep_creates_all_combinations(expander, namer):
    """Expand sweep mode into all parameter combinations."""
    data = {
        "sweep": [
            {
                "name_prefix": "tsp_test",
                "runs": 10,
                "seed_base": 99,
                "problem": {"file_path": "p.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [100, 200],
                    "crossover_rate": [0.8, 0.9],
                    "mutation_rate": 0.05,
                    "max_time": 10.0,
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 4
    assert all(isinstance(r, ExperimentConfig) for r in result)
    assert all(r.runs == 10 for r in result)
    assert all(r.seed_base == 99 for r in result)
    assert len(namer.calls) == 4
    assert {"tsp_test_100", "tsp_test_200"} <= {r.name for r in result}


def test_expand_sweep_validates_problem_and_algorithm(expander, validator):
    """Ensure sweep mode triggers proper validation calls."""
    data = {
        "sweep": [
            {
                "name_prefix": "abc",
                "runs": 1,
                "seed_base": 7,
                "problem": {"file_path": "x.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [10],
                    "crossover_rate": [0.5],
                    "mutation_rate": 0.01,
                    "max_time": 5,
                },
            }
        ]
    }
    expander.expand(data)
    assert validator.validated_problems == [
        {"file_path": "x.tsp", "optimal_results_path": "opt.json"}
    ]
    assert len(validator.validated_algorithms) == 1
    assert validator.validated_algorithms[0][1] is False


def test_expand_sweep_multiple_param_lists(expander):
    """Handle multi-dimensional Cartesian product correctly."""
    data = {
        "sweep": [
            {
                "name_prefix": "xyz",
                "runs": 3,
                "seed_base": 9,
                "problem": {"file_path": "z.tsp", "optimal_results_path": "opt.json"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [50, 100],
                    "crossover_rate": [0.7, 0.8],
                    "mutation_rate": [0.05, 0.1],
                    "max_time": 5.0,
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 8
    assert all(r.name.startswith("xyz") for r in result)


def test_expand_with_no_valid_section_logs_warning(expander, caplog):
    """Log warning if no valid experiment section exists."""
    data = {"invalid": []}
    with caplog.at_level("WARNING"):
        result = expander.expand(data)
    assert result == []
    assert "No valid experiment section" in caplog.text


def test_expand_empty_experiments_returns_empty_list(expander):
    """Return empty list for empty experiments block."""
    data = {"experiments": []}
    result = expander.expand(data)
    assert result == []
