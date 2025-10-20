import pytest

from src.core.name_generator import NameGenerator


@pytest.fixture
def base_problem():
    return {"file_path": "data/tsplib/ulysses16.tsp"}


@pytest.fixture
def base_algorithm():
    return {
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
    gen = NameGenerator()
    result = gen.generate(base_problem, base_algorithm, "prefix")
    assert result.startswith("prefix_tsp_ulysses16_ga_population_100_")
    assert "_elitist_0_2" in result
    assert "tournament_0_1" in result
    assert "ox_0_9_insert_0_05" in result


def test_generate_with_replacement_rate(base_problem, base_algorithm):
    alg = base_algorithm.copy()
    alg["succession_config"] = {"name": "steady_state", "replacement_rate": 0.15}
    gen = NameGenerator()
    result = gen.generate(base_problem, alg, "pref")
    assert "_steady_state_0_15" in result
    assert "pref_tsp_ulysses16_ga_population_100_" in result


def test_generate_with_no_rate_fields(base_problem, base_algorithm):
    alg = base_algorithm.copy()
    alg["succession_config"] = {"name": "steady_state"}  # brak rate
    gen = NameGenerator()
    result = gen.generate(base_problem, alg, "run")
    assert result.endswith("_steady_state_0")
    assert "_0" in result


def test_generate_with_different_selection_rate_types(base_problem, base_algorithm):
    alg = base_algorithm.copy()
    alg["selection_config"]["rate"] = 0.05
    gen = NameGenerator()
    result = gen.generate(base_problem, alg, "tsp_experiment")
    assert "tournament_0_05" in result
    assert "_ox_0_9_insert_0_05_" in result


def test_generate_with_different_prefix_and_instance(tmp_path):
    path = tmp_path / "berlin52.tsp"
    path.write_text("dummy", encoding="utf-8")

    problem = {"file_path": str(path)}
    alg = {
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
    result = gen.generate(problem, alg, "exp_set")
    assert result.startswith("exp_set_tsp_berlin52_ga_population_500_")
    assert "_pmx_0_8_swap_0_1_elitist_0_05" in result


def test_generate_uses_int_time_value(base_problem, base_algorithm):
    alg = base_algorithm.copy()
    alg["max_time"] = 5.9
    gen = NameGenerator()
    result = gen.generate(base_problem, alg, "exp")
    assert "_time_5_" in result
