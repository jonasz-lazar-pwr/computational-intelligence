import pytest

from src.core.config_validator import ConfigValidator


@pytest.fixture
def validator():
    """Return ConfigValidator."""
    return ConfigValidator()


def test_validate_root_accepts_valid_experiments(validator, caplog):
    """Validate root with 'experiments' section."""
    data = {"experiments": []}
    with caplog.at_level("DEBUG"):
        validator.validate_root(data)
    assert "Root configuration validated successfully." in caplog.text


def test_validate_root_accepts_valid_sweep(validator):
    """Validate root with 'sweep' section."""
    validator.validate_root({"sweep": []})


@pytest.mark.parametrize("data", [None, [], "string", 123])
def test_validate_root_raises_if_not_dict(validator, data):
    """Raise if root is not a mapping."""
    with pytest.raises(ValueError, match="Top-level YAML must be a mapping"):
        validator.validate_root(data)


def test_validate_root_raises_if_missing_sections(validator):
    """Raise if root lacks required sections."""
    with pytest.raises(ValueError, match="YAML must contain either 'experiments' or 'sweep'"):
        validator.validate_root({"invalid": []})


def test_validate_problem_accepts_valid(validator, caplog):
    """Validate valid problem section."""
    problem = {"file_path": "a.tsp", "optimal_results_path": "opt.json"}
    with caplog.at_level("DEBUG"):
        validator.validate_problem(problem)
    assert "Problem configuration validated successfully." in caplog.text


@pytest.mark.parametrize(
    "problem",
    [{}, {"file_path": "only.tsp"}, {"optimal_results_path": "only.json"}],
)
def test_validate_problem_raises_on_missing_fields(validator, problem):
    """Raise when problem fields missing."""
    with pytest.raises(ValueError, match="Problem section must contain"):
        validator.validate_problem(problem)


@pytest.fixture
def valid_ga():
    """Return minimal valid GA config."""
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


def test_validate_algorithm_accepts_valid_ga(validator, valid_ga, caplog):
    """Validate correct GA configuration."""
    with caplog.at_level("DEBUG"):
        validator.validate_algorithm(valid_ga, allow_lists=False)
    assert "Algorithm configuration validated successfully." in caplog.text


@pytest.mark.parametrize(
    "missing_key",
    ["population_size", "crossover_rate", "mutation_rate", "max_time"],
)
def test_validate_algorithm_ga_missing_required_field(validator, valid_ga, missing_key):
    """Raise for missing GA required scalar fields."""
    del valid_ga[missing_key]
    with pytest.raises(ValueError, match=f"Missing required algorithm field: {missing_key}"):
        validator.validate_algorithm(valid_ga, allow_lists=False)


@pytest.mark.parametrize(
    "missing_section",
    ["selection_config", "crossover_config", "mutation_config", "succession_config"],
)
def test_validate_algorithm_ga_missing_operator_sections(validator, valid_ga, missing_section):
    """Raise for missing GA operator sections."""
    del valid_ga[missing_section]
    with pytest.raises(
        ValueError, match=f"Algorithm configuration must include {missing_section} section"
    ):
        validator.validate_algorithm(valid_ga, allow_lists=False)


def test_validate_algorithm_ga_rejects_list_when_not_allowed(validator, valid_ga):
    """Reject lists when allow_lists=False."""
    valid_ga["population_size"] = [10, 20]
    with pytest.raises(ValueError, match="Unexpected list"):
        validator.validate_algorithm(valid_ga, allow_lists=False)


def test_validate_algorithm_ga_allows_list_in_sweep(validator, valid_ga):
    """Allow lists when allow_lists=True."""
    valid_ga["population_size"] = [10, 20]
    validator.validate_algorithm(valid_ga, allow_lists=True)


@pytest.fixture
def valid_acs():
    """Return minimal valid ACS config."""
    return {
        "name": "acs",
        "num_ants": 5,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "phi": 0.05,
        "q0": 0.8,
        "max_time": 10,
    }


def test_validate_algorithm_accepts_valid_acs(validator, valid_acs, caplog):
    """Validate correct ACS configuration."""
    with caplog.at_level("DEBUG"):
        validator.validate_algorithm(valid_acs, allow_lists=False)
    assert "Algorithm configuration validated successfully." in caplog.text


@pytest.mark.parametrize(
    "missing_key",
    ["num_ants", "alpha", "beta", "rho", "phi", "q0", "max_time"],
)
def test_validate_algorithm_acs_missing_required_fields(validator, valid_acs, missing_key):
    """Raise for missing ACS required fields."""
    del valid_acs[missing_key]
    with pytest.raises(ValueError, match=f"Missing required ACS field: {missing_key}"):
        validator.validate_algorithm(valid_acs, allow_lists=False)


def test_validate_algorithm_acs_rejects_unknown_fields(validator, valid_acs):
    """Reject GA-only fields in ACS."""
    valid_acs["population_size"] = 10
    with pytest.raises(ValueError, match="Unexpected GA field in ACS"):
        validator.validate_algorithm(valid_acs, allow_lists=False)


def test_validate_algorithm_acs_allows_lists_in_sweep(validator, valid_acs):
    """Allow lists in ACS when allow_lists=True."""
    valid_acs["num_ants"] = [5, 10]
    validator.validate_algorithm(valid_acs, allow_lists=True)
