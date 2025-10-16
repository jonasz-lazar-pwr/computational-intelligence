from pathlib import Path

import pytest

from src.problems.tsp.tsp_instance import TSPInstance
from src.problems.tsp.tsp_problem import TSPProblem


@pytest.fixture()
def mock_tsp_instance(tmp_path: Path) -> TSPInstance:
    """Return mock TSPInstance with small predefined distance matrix."""
    tsp_file = tmp_path / "mock.tsp"
    tsp_file.write_text(
        """NAME: mock
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 2 9
2 0 6
9 6 0
EOF
""",
        encoding="utf-8",
    )
    instance = TSPInstance(str(tsp_file), optimal_results_path=str(tmp_path / "optimal.json"))
    instance.name = "mock"
    instance.dimension = 3
    instance.has_loaded = True
    instance.distance_matrix = [[0, 2, 9], [2, 0, 6], [9, 6, 0]]
    instance.optimal_result = 17
    instance.edge_weight_type = "EXPLICIT"
    instance.parser.file_path = tsp_file
    return instance


@pytest.fixture()
def tsp_problem(mock_tsp_instance: TSPInstance) -> TSPProblem:
    """Provide TSPProblem adapter based on mock instance."""
    return TSPProblem(mock_tsp_instance)


def test_initialization_loads_matrix(tsp_problem: TSPProblem):
    """Verify instance linkage and correct dimension."""
    assert tsp_problem.instance is not None
    assert tsp_problem.get_dimension() == tsp_problem.instance.dimension


def test_evaluate_valid_solution(tsp_problem: TSPProblem):
    """Verify correct tour distance calculation."""
    result = tsp_problem.evaluate([0, 1, 2])
    assert result == pytest.approx(17.0)


def test_evaluate_raises_if_no_matrix(mock_tsp_instance: TSPInstance):
    """Verify error raised when matrix missing."""
    mock_tsp_instance.has_loaded = False
    mock_tsp_instance.distance_matrix = []
    problem = TSPProblem(mock_tsp_instance)
    assert problem.instance.has_loaded
    assert problem.instance.distance_matrix


def test_get_initial_solution_returns_sequential(tsp_problem: TSPProblem):
    """Verify sequential initial solution."""
    solution = tsp_problem.get_initial_solution()
    assert solution == [0, 1, 2]


def test_optimal_value_returns_instance_value(tsp_problem: TSPProblem):
    """Verify optimal value matches instance value."""
    assert tsp_problem.optimal_value() == 17


def test_info_contains_problem_metadata(tsp_problem: TSPProblem):
    """Verify metadata includes base and TSP details."""
    info = tsp_problem.info()
    assert info["name"] == "mock"
    assert info["dimension"] == 3
    assert info["edge_weight_type"] == "EXPLICIT"
    assert info["optimal_result"] == 17
