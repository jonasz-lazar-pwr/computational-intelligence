import pytest

from src.core.config_expander import ConfigExpander
from src.core.models import ExperimentConfig
from src.interfaces.core_interfaces import IConfigValidator, INameGenerator


class DummyValidator(IConfigValidator):
    """Record validation calls."""

    def __init__(self):
        self.validated_problems = []
        self.validated_algorithms = []

    def validate_root(self, data): ...
    def validate_problem(self, problem):
        self.validated_problems.append(problem)

    def validate_algorithm(self, algorithm, allow_lists):
        self.validated_algorithms.append((algorithm, allow_lists))


class DummyNamer(INameGenerator):
    """Generate deterministic experiment names."""

    def __init__(self):
        self.calls = []

    def generate(self, problem, alg):
        """Return deterministic name based on algorithm type."""
        algo = alg.get("name", "unknown")
        key = alg.get("population_size") if algo == "ga" else alg.get("num_ants")
        name = f"{problem.get('name', 'p')}_{problem.get('instance_name', 'x')}_{algo}_{key}"
        self.calls.append((problem, alg))
        return name


@pytest.fixture
def expander():
    """Return ConfigExpander instance using dummy validator and namer."""
    return ConfigExpander(DummyValidator(), DummyNamer())


def test_expand_manual_single(expander):
    """Expand single manual experiment."""
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
    assert cfg.algorithm["population_size"] == 100


def test_expand_manual_multiple(expander):
    """Expand two manual experiment entries."""
    data = {
        "experiments": [
            {
                "name": "A",
                "runs": 1,
                "seed_base": 1,
                "problem": {"name": "tsp", "instance_name": "berlin52"},
                "algorithm": {"name": "ga", "population_size": 50},
            },
            {
                "name": "B",
                "runs": 2,
                "seed_base": 2,
                "problem": {"name": "tsp", "instance_name": "eil51"},
                "algorithm": {"name": "ga", "population_size": 80},
            },
        ]
    }
    result = expander.expand(data)
    assert [r.name for r in result] == ["A", "B"]


def test_expand_sweep_basic_ga(expander):
    """Expand GA sweep of two list parameters."""
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
                    "selection_config": [{"name": "tournament"}],
                    "crossover_config": [{"name": "ox"}],
                    "mutation_config": [{"name": "swap"}],
                    "succession_config": [{"name": "elitist", "elite_rate": 0.1}],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 4
    assert all(r.runs == 3 for r in result)
    assert all(r.seed_base == 9 for r in result)


def test_expand_operator_section_with_lists_ga(expander):
    """Expand GA operators with nested lists."""
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
                    "crossover_rate": [0.8],
                    "mutation_rate": [0.1],
                    "max_time": [1],
                    "selection_config": [
                        {"name": "tournament", "rate": [0.05, 0.1]},
                        {"name": "roulette"},
                    ],
                    "crossover_config": [{"name": "ox"}],
                    "mutation_config": [{"name": "swap"}],
                    "succession_config": [{"name": "elitist", "elite_rate": 0.1}],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 3
    assert all(isinstance(r, ExperimentConfig) for r in result)


def test_expand_sweep_deduplicates_ga(expander):
    """Deduplicate GA configurations."""
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
                    "mutation_rate": 0.05,
                    "max_time": 5,
                    "selection_config": [{"name": "tournament"}],
                    "crossover_config": [{"name": "ox"}],
                    "mutation_config": [{"name": "swap"}],
                    "succession_config": [{"name": "elitist", "elite_rate": 0.1}],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 1


def test_expand_sweep_basic_acs(expander):
    """Expand ACS sweep of varying parameters."""
    data = {
        "sweep": [
            {
                "name": "acs_sweep",
                "runs": 2,
                "seed_base": 7,
                "problem": {"name": "tsp", "instance_name": "test"},
                "algorithm": {
                    "name": "acs",
                    "num_ants": [5, 10],
                    "alpha": [1.0],
                    "beta": [2.0],
                    "rho": [0.1, 0.2],
                    "phi": [0.05],
                    "q0": [0.8],
                    "max_time": [5],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 4
    assert all(r.algorithm["name"] == "acs" for r in result)


def test_expand_sweep_deduplicates_acs(expander):
    """Deduplicate ACS scalar configurations."""
    data = {
        "sweep": [
            {
                "name": "acs_dup",
                "runs": 1,
                "seed_base": 2,
                "problem": {"name": "tsp", "instance_name": "dup"},
                "algorithm": {
                    "name": "acs",
                    "num_ants": [10, 10],
                    "alpha": [1.0],
                    "beta": [2.0],
                    "rho": [0.1],
                    "phi": [0.05],
                    "q0": [0.8],
                    "max_time": [5],
                },
            }
        ]
    }
    result = expander.expand(data)
    assert len(result) == 1


def test_expand_invalid_section_logs_warning(expander, caplog):
    """Log warning when no valid section exists."""
    data = {"foo": []}
    with caplog.at_level("WARNING"):
        result = expander.expand(data)
    assert result == []
    assert "No valid experiment section" in caplog.text


def test_expand_empty_experiments_returns_empty(expander):
    """Return empty list when 'experiments' is empty."""
    assert expander.expand({"experiments": []}) == []
