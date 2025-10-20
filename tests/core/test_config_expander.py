import pytest

from src.core.config_expander import ConfigExpander
from src.core.models import ExperimentConfig
from src.interfaces.core_interfaces import IConfigValidator, INameGenerator


class DummyValidator(IConfigValidator):
    """Simple mock validator to track validation calls."""

    def __init__(self):
        self.validated_problems = []
        self.validated_algorithms = []

    def validate_root(self, data): ...
    def validate_problem(self, problem):
        self.validated_problems.append(problem)

    def validate_algorithm(self, algorithm, allow_lists):
        self.validated_algorithms.append((algorithm, allow_lists))


class DummyNamer(INameGenerator):
    """Simple mock namer for reproducible results."""

    def __init__(self):
        self.calls = []

    def generate(self, problem, alg):
        """Return a predictable name based on problem + population."""
        name = f"{problem.get('name', 'p')}_{problem.get('instance_name', 'x')}_{alg.get('population_size', 'X')}"
        self.calls.append((problem, alg))
        return name


@pytest.fixture
def expander():
    """Provide ConfigExpander with dummy validator and namer."""
    return ConfigExpander(DummyValidator(), DummyNamer())


def test_expand_manual_single(expander):
    """Ensure single manual experiment expands correctly."""
    data = {
        "experiments": [
            {
                "name": "exp1",
                "runs": 5,
                "seed_base": 42,
                "problem": {"name": "tsp", "instance_name": "ulysses16"},
                "algorithm": {"name": "ga", "population_size": 100},
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 1
    cfg = result[0]
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "exp1"
    assert cfg.runs == 5
    assert cfg.seed_base == 42
    assert "instance_name" in cfg.problem
    assert cfg.algorithm["population_size"] == 100


def test_expand_manual_multiple(expander):
    """Two manual definitions produce two distinct ExperimentConfig objects."""
    data = {
        "experiments": [
            {
                "name": "A",
                "runs": 1,
                "seed_base": 1,
                "problem": {"name": "tsp", "instance_name": "berlin52"},
                "algorithm": {"name": "ga"},
            },
            {
                "name": "B",
                "runs": 2,
                "seed_base": 2,
                "problem": {"name": "tsp", "instance_name": "eil51"},
                "algorithm": {"name": "ga"},
            },
        ]
    }
    result = expander.expand(data)
    assert [r.name for r in result] == ["A", "B"]
    assert all(isinstance(r, ExperimentConfig) for r in result)


def test_expand_sweep_basic(expander):
    """Expand 2x2 parameter grid for sweep experiments."""
    data = {
        "sweep": [
            {
                "name": "ga_sweep",
                "runs": 3,
                "seed_base": 9,
                "problem": {"name": "tsp", "instance_name": "test"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [100, 200],
                    "crossover_rate": [0.8, 0.9],
                    "mutation_rate": 0.05,
                    "max_time": 10,
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 4
    assert all(isinstance(r, ExperimentConfig) for r in result)
    assert all(r.runs == 3 for r in result)
    assert all(r.seed_base == 9 for r in result)


def test_expand_operator_section_with_lists(expander):
    """Operator configs with lists should expand to all valid combinations."""
    data = {
        "sweep": [
            {
                "name": "optest",
                "runs": 1,
                "seed_base": 1,
                "problem": {"name": "tsp", "instance_name": "dummy"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [100],
                    "selection_config": [
                        {"name": "tournament", "rate": [0.05, 0.1]},
                        {"name": "roulette"},
                    ],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 3
    assert all(isinstance(r, ExperimentConfig) for r in result)


def test_expand_sweep_deduplicates(expander):
    """Ensure duplicate algorithm configurations are removed."""
    data = {
        "sweep": [
            {
                "name": "dedup",
                "runs": 1,
                "seed_base": 1,
                "problem": {"name": "tsp", "instance_name": "dup"},
                "algorithm": {
                    "name": "ga",
                    "population_size": [100, 100],
                    "crossover_rate": [0.8],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 1


def test_expand_invalid_section_logs_warning(expander, caplog):
    """Warn and return empty list if no valid experiment section exists."""
    data = {"foo": []}
    with caplog.at_level("WARNING"):
        result = expander.expand(data)
    assert result == []
    assert "No valid experiment section" in caplog.text


def test_expand_empty_experiments_returns_empty(expander):
    """Empty 'experiments' section should return an empty list."""
    result = expander.expand({"experiments": []})
    assert result == []
