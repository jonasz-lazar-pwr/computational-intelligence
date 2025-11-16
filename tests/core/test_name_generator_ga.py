import pytest

from src.core.name_generator import NameGenerator


@pytest.fixture
def base_problem():
    """Return base problem config."""
    return {
        "name": "tsp",
        "instance_name": "ulysses16",
        "file_path": "data/tsplib/ulysses16.tsp",
    }


@pytest.fixture
def base_algorithm():
    """Return base GA algorithm config."""
    return {
        "name": "ga",
        "population_size": 100,
        "max_time": 10.0,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "selection_config": {"name": "tournament", "rate": 0.1},
        "crossover_config": {"name": "ox"},
        "mutation_config": {"name": "insert"},
        "succession_config": {"name": "elitist", "elite_rate": 0.2},
    }


def test_generate_with_elite_rate(base_problem, base_algorithm):
    """Generate GA name containing elite_rate."""
    gen = NameGenerator()
    result = gen.generate(base_problem, base_algorithm)

    assert result.startswith("tsp_ulysses16_ga_population_100_time_10_")
    assert "_tournament_0_1_" in result
    assert "_ox_0_9_" in result
    assert "_insert_0_05_" in result
    assert "_elitist_0_2" in result


def test_generate_with_replacement_rate(base_problem, base_algorithm):
    """Generate GA name with replacement_rate instead of elite_rate."""
    alg = base_algorithm.copy()
    alg["succession_config"] = {"name": "steady_state", "replacement_rate": 0.15}

    gen = NameGenerator()
    result = gen.generate(base_problem, alg)

    assert result.startswith("tsp_ulysses16_ga_population_100_time_10_")
    assert "_steady_state_0_15" in result


def test_generate_with_no_rate_fields(base_problem, base_algorithm):
    """Ensure default rate is 0 when elite/replacement rate is missing."""
    alg = base_algorithm.copy()
    alg["succession_config"] = {"name": "steady_state"}

    gen = NameGenerator()
    result = gen.generate(base_problem, alg)

    assert result.endswith("_steady_state_0")
    assert "_steady_state_0" in result


def test_generate_with_different_selection_rate_types(base_problem, base_algorithm):
    """Ensure selection rate is sanitized correctly."""
    alg = base_algorithm.copy()
    alg["selection_config"]["rate"] = 0.05

    gen = NameGenerator()
    result = gen.generate(base_problem, alg)

    assert "tournament_0_05" in result
    assert "_ox_0_9_" in result
    assert "_insert_0_05_" in result


def test_generate_with_different_instance_name(tmp_path):
    """Generate GA name for a different TSP instance."""
    path = tmp_path / "berlin52.tsp"
    path.write_text("dummy", encoding="utf-8")

    problem = {
        "name": "tsp",
        "instance_name": "berlin52",
        "file_path": str(path),
    }
    alg = {
        "name": "ga",
        "population_size": 500,
        "max_time": 30.0,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "selection_config": {"name": "rank"},
        "crossover_config": {"name": "pmx"},
        "mutation_config": {"name": "swap"},
        "succession_config": {"name": "elitist", "elite_rate": 0.05},
    }

    gen = NameGenerator()
    result = gen.generate(problem, alg)

    assert result.startswith("tsp_berlin52_ga_population_500_time_30_")
    assert "_rank_0_" in result
    assert "_pmx_0_8_" in result
    assert "_swap_0_1_" in result
    assert "_elitist_0_05" in result


def test_generate_uses_int_time_value(base_problem, base_algorithm):
    """Ensure max_time is converted to integer."""
    alg = base_algorithm.copy()
    alg["max_time"] = 5.9

    gen = NameGenerator()
    result = gen.generate(base_problem, alg)

    assert "_time_5_" in result
    assert result.startswith("tsp_ulysses16_ga_population_100_")
    assert "elitist" in result
