import pytest

from src.core.config_validator import ConfigValidator


@pytest.fixture
def validator():
    """Provide ConfigValidator instance."""
    return ConfigValidator()


def test_validate_root_accepts_valid_experiments(validator, caplog):
    """Accept valid root with experiments section."""
    data = {"experiments": []}
    with caplog.at_level("DEBUG"):
        validator.validate_root(data)
    assert "Root configuration validated successfully." in caplog.text


def test_validate_root_accepts_valid_sweep(validator):
    """Accept valid root with sweep section."""
    data = {"sweep": []}
    validator.validate_root(data)


@pytest.mark.parametrize("data", [None, [], "string", 123])
def test_validate_root_raises_if_not_dict(validator, data):
    """Raise ValueError if root is not a dict."""
    with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
        validator.validate_root(data)


def test_validate_root_raises_if_missing_sections(validator):
    """Raise ValueError if required root keys are missing."""
    with pytest.raises(ValueError, match="YAML must contain either 'experiments' or 'sweep'"):
        validator.validate_root({"invalid": []})


def test_validate_problem_accepts_valid(validator, caplog):
    """Accept valid problem section."""
    problem = {"file_path": "path.tsp", "optimal_results_path": "opt.json"}
    with caplog.at_level("DEBUG"):
        validator.validate_problem(problem)
    assert "Problem configuration validated successfully." in caplog.text


@pytest.mark.parametrize(
    "problem",
    [
        {},
        {"file_path": "only_path.tsp"},
        {"optimal_results_path": "only_opt.json"},
    ],
)
def test_validate_problem_raises_on_missing_fields(validator, problem):
    """Raise ValueError if problem fields are missing."""
    with pytest.raises(
        ValueError,
        match="Problem section must contain 'file_path' and 'optimal_results_path'",
    ):
        validator.validate_problem(problem)


@pytest.fixture
def valid_algorithm():
    """Provide minimal valid algorithm configuration."""
    return {
        "name": "ga",
        "population_size": 100,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "max_time": 10,
        "selection_config": {"name": "tournament"},
        "crossover_config": {"name": "ox"},
        "mutation_config": {"name": "insert"},
        "succession_config": {"name": "elitist"},
    }


def test_validate_algorithm_accepts_valid(validator, valid_algorithm, caplog):
    """Accept valid algorithm configuration."""
    with caplog.at_level("DEBUG"):
        validator.validate_algorithm(valid_algorithm, allow_lists=False)
    assert "Algorithm configuration validated successfully." in caplog.text


@pytest.mark.parametrize(
    "missing_key",
    ["name", "population_size", "crossover_rate", "mutation_rate", "max_time"],
)
def test_validate_algorithm_raises_on_missing_required_field(
    validator, valid_algorithm, missing_key
):
    """Raise ValueError if required algorithm field is missing."""
    del valid_algorithm[missing_key]
    with pytest.raises(ValueError, match=f"Missing required algorithm field: {missing_key}"):
        validator.validate_algorithm(valid_algorithm, allow_lists=False)


@pytest.mark.parametrize(
    "missing_section",
    ["selection_config", "crossover_config", "mutation_config", "succession_config"],
)
def test_validate_algorithm_raises_on_missing_subconfig(
    validator, valid_algorithm, missing_section
):
    """Raise ValueError if required subconfig is missing."""
    del valid_algorithm[missing_section]
    with pytest.raises(
        ValueError, match=f"Algorithm configuration must include {missing_section} section"
    ):
        validator.validate_algorithm(valid_algorithm, allow_lists=False)


def test_validate_algorithm_raises_on_list_when_not_allowed(validator, valid_algorithm):
    """Raise ValueError if list values not allowed outside sweep mode."""
    valid_algorithm["population_size"] = [100, 200]
    with pytest.raises(
        ValueError, match="Unexpected list in algorithm config for field: population_size"
    ):
        validator.validate_algorithm(valid_algorithm, allow_lists=False)


def test_validate_algorithm_allows_list_in_sweep_mode(validator, valid_algorithm):
    """Allow list fields when allow_lists=True."""
    valid_algorithm["population_size"] = [100, 200]
    validator.validate_algorithm(valid_algorithm, allow_lists=True)
