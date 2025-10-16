import pytest

from src.problems.base_problem import BaseProblem


@pytest.fixture
def base_problem() -> BaseProblem:
    """Return a simple BaseProblem instance for testing."""
    return BaseProblem(name="TestProblem", dimension=5)


def test_initialization(base_problem: BaseProblem) -> None:
    """Check initialization sets correct attributes."""
    assert base_problem.name == "TestProblem"
    assert base_problem.dimension == 5


def test_get_dimension(base_problem: BaseProblem) -> None:
    """get_dimension returns the correct dimension."""
    assert base_problem.get_dimension() == 5


def test_get_initial_solution(base_problem: BaseProblem) -> None:
    """Initial solution should be a sequential list from 0 to n-1."""
    solution = base_problem.get_initial_solution()
    assert solution == [0, 1, 2, 3, 4]
    assert len(solution) == base_problem.get_dimension()


def test_info_contains_expected_fields(base_problem: BaseProblem) -> None:
    """info() should return correct name and dimension."""
    info = base_problem.info()
    assert info["name"] == "TestProblem"
    assert info["dimension"] == 5
    assert set(info.keys()) == {"name", "dimension"}


def test_optimal_value_returns_none(base_problem: BaseProblem) -> None:
    """Default optimal value should be None."""
    assert base_problem.optimal_value() is None
