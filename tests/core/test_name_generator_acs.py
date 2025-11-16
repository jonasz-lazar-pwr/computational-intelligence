import pytest

from src.core.name_generator import NameGenerator


@pytest.fixture
def base_problem():
    """Return base problem config."""
    return {
        "name": "tsp",
        "instance_name": "berlin52",
        "file_path": "data/tsplib/berlin52.tsp",
    }


@pytest.fixture
def base_acs_algorithm():
    """Return base ACS algorithm config."""
    return {
        "name": "acs",
        "num_ants": 20,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "phi": 0.05,
        "q0": 0.9,
        "max_time": 10.0,
    }


def test_generate_basic_acs_name(base_problem, base_acs_algorithm):
    """Generate basic ACS name and verify components."""
    gen = NameGenerator()
    name = gen.generate(base_problem, base_acs_algorithm)

    assert name.startswith("tsp_berlin52_acs_")
    assert "_ants_20_" in name
    assert "_alpha_1_" in name
    assert "_beta_2_" in name
    assert "_rho_0_1_" in name
    assert "_phi_0_05_" in name
    assert "_q0_0_9_" in name
    assert name.endswith("_time_10")


def test_generate_acs_time_sanitization(base_problem, base_acs_algorithm):
    """Ensure float max_time is converted to int."""
    alg = base_acs_algorithm.copy()
    alg["max_time"] = 19.7

    gen = NameGenerator()
    name = gen.generate(base_problem, alg)

    assert name.endswith("_time_19")


def test_generate_acs_sanitizes_all_float_values(base_problem, base_acs_algorithm):
    """Verify float parameters are sanitized correctly."""
    alg = base_acs_algorithm.copy()
    alg["alpha"] = 0.33
    alg["beta"] = 1.75
    alg["rho"] = 0.99
    alg["phi"] = 0.123
    alg["q0"] = 0.555

    gen = NameGenerator()
    name = gen.generate(base_problem, alg)

    assert "_alpha_0_33_" in name
    assert "_beta_1_75_" in name
    assert "_rho_0_99_" in name
    assert "_phi_0_123_" in name
    assert "_q0_0_555_" in name


def test_generate_acs_handles_integer_like_values(base_problem, base_acs_algorithm):
    """Verify integer-like values are preserved cleanly."""
    alg = base_acs_algorithm.copy()
    alg["alpha"] = 2
    alg["beta"] = 4
    alg["rho"] = 1
    alg["phi"] = 0
    alg["q0"] = 1

    gen = NameGenerator()
    name = gen.generate(base_problem, alg)

    assert "_alpha_2_" in name
    assert "_beta_4_" in name
    assert "_rho_1_" in name
    assert "_phi_0_" in name
    assert "_q0_1_" in name


def test_generate_acs_different_instance_name(tmp_path, base_acs_algorithm):
    """Generate ACS name for another instance_name."""
    file = tmp_path / "ulysses22.tsp"
    file.write_text("dummy", encoding="utf-8")

    problem = {
        "name": "tsp",
        "instance_name": "ulysses22",
        "file_path": str(file),
    }

    gen = NameGenerator()
    name = gen.generate(problem, base_acs_algorithm)

    assert name.startswith("tsp_ulysses22_acs_")
    assert "_ants_20_" in name
    assert name.endswith("_time_10")


def test_generate_acs_missing_required_field_raises(base_problem, base_acs_algorithm):
    """Missing ACS parameters must raise ValueError."""
    gen = NameGenerator()

    for key in ["num_ants", "alpha", "beta", "rho", "phi", "q0", "max_time"]:
        alg = base_acs_algorithm.copy()
        alg.pop(key)

        with pytest.raises(ValueError, match=f"Missing required algorithm parameter: '{key}'"):
            gen.generate(base_problem, alg)


def test_generate_acs_rejects_unknown_algorithm_type():
    """Unknown algorithm type must raise ValueError."""
    gen = NameGenerator()
    problem = {"name": "tsp", "instance_name": "x"}
    alg = {"name": "xxx"}

    with pytest.raises(ValueError):
        gen.generate(problem, alg)
