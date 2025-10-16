import json

from pathlib import Path

import pytest

from src.problems.tsp.tsp_instance import TSPInstance


@pytest.fixture()
def valid_tsp_file(tmp_path: Path) -> Path:
    """Return valid EUC_2D TSP file."""
    content = """NAME: berlin52
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10 10
2 20 10
3 20 20
EOF
"""
    file_path = tmp_path / "berlin52.tsp"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture()
def optimal_results_path(tmp_path: Path) -> Path:
    """Return temporary JSON with optimal result."""
    data = {"berlin52": 7542}
    file_path = tmp_path / "optimal.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


@pytest.fixture()
def tsp_instance(valid_tsp_file: Path, optimal_results_path: Path) -> TSPInstance:
    """Return initialized TSPInstance for testing."""
    return TSPInstance(str(valid_tsp_file), str(optimal_results_path))


def test_load_metadata_success(tsp_instance: TSPInstance):
    """Test successful metadata loading."""
    tsp_instance.load_metadata()
    assert tsp_instance.name == "berlin52"
    assert tsp_instance.dimension == 3
    assert tsp_instance.edge_weight_type == "EUC_2D"
    assert tsp_instance.optimal_result == 7542


def test_to_dict_after_load(tsp_instance: TSPInstance):
    """Test conversion to dict after metadata load."""
    tsp_instance.load_metadata()
    d = tsp_instance.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == "berlin52"
    assert d["has_loaded"] is False


def test_load_metadata_missing_file(optimal_results_path: Path):
    """Test missing TSP file raises error."""
    inst = TSPInstance("nonexistent_file.tsp", str(optimal_results_path))
    with pytest.raises(FileNotFoundError):
        inst.load_metadata()


def test_load_metadata_unexpected_error(monkeypatch, tsp_instance: TSPInstance):
    """Test unexpected parser error."""

    def bad_validate_file(self, path):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr("src.problems.tsp.tsp_parser.TSPParser.validate_file", bad_validate_file)
    with pytest.raises(RuntimeError):
        tsp_instance.load_metadata()


def test_load_optimal_result_file_missing(valid_tsp_file: Path):
    """Test missing optimal.json does not raise error."""
    inst = TSPInstance(str(valid_tsp_file), "missing.json")
    inst._load_optimal_result()
    assert inst.optimal_result is None


def test_load_optimal_result_invalid_json(valid_tsp_file: Path, tmp_path: Path):
    """Test invalid JSON format."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid", encoding="utf-8")
    inst = TSPInstance(str(valid_tsp_file), str(bad_json))
    inst._load_optimal_result()
    assert inst.optimal_result is None


def test_load_optimal_result_not_dict(valid_tsp_file: Path, tmp_path: Path):
    """Test JSON content not being a dict."""
    not_dict = tmp_path / "list.json"
    not_dict.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    inst = TSPInstance(str(valid_tsp_file), str(not_dict))
    inst._load_optimal_result()
    assert inst.optimal_result is None


def test_load_optimal_result_no_key(valid_tsp_file: Path, tmp_path: Path):
    """Test JSON missing expected key."""
    missing_key = tmp_path / "opt.json"
    missing_key.write_text(json.dumps({"other": 1234}), encoding="utf-8")
    inst = TSPInstance(str(valid_tsp_file), str(missing_key))
    inst._load_optimal_result()
    assert inst.optimal_result is None


def test_load_optimal_result_valid(valid_tsp_file: Path, optimal_results_path: Path):
    """Test valid optimal.json loading."""
    inst = TSPInstance(str(valid_tsp_file), str(optimal_results_path))
    inst._load_optimal_result()
    assert inst.optimal_result == 7542


def test_load_display_coordinates(monkeypatch, tsp_instance: TSPInstance):
    """Test display coordinate loading."""
    monkeypatch.setattr(tsp_instance.parser, "load_display_coordinates", lambda: [(1.0, 2.0)])
    tsp_instance.load_display_coordinates()
    assert tsp_instance.display_coordinates == [(1.0, 2.0)]


def test_load_display_coordinates_fails(monkeypatch, tsp_instance: TSPInstance):
    """Test failure in display coordinate loading."""

    def raise_error():
        raise ValueError("bad display")

    monkeypatch.setattr(tsp_instance.parser, "load_display_coordinates", raise_error)
    tsp_instance.load_display_coordinates()
    assert tsp_instance.display_coordinates == []


def test_load_distance_matrix_first_and_second(monkeypatch, tsp_instance: TSPInstance):
    """Test distance matrix loading and reload."""
    monkeypatch.setattr(tsp_instance.parser, "generate_distance_matrix", lambda: None)
    monkeypatch.setattr(tsp_instance.parser, "get_distance_matrix", lambda: [[0, 1], [1, 0]])
    tsp_instance.load_distance_matrix()
    assert tsp_instance.has_loaded is True
    assert tsp_instance.distance_matrix == [[0, 1], [1, 0]]
    tsp_instance.load_distance_matrix()


def test_load_distance_matrix_fails(monkeypatch, tsp_instance: TSPInstance):
    """Test exception during distance matrix generation."""

    def raise_error():
        raise RuntimeError("fail gen")

    monkeypatch.setattr(tsp_instance.parser, "generate_distance_matrix", raise_error)
    with pytest.raises(RuntimeError):
        tsp_instance.load_distance_matrix()


def test_get_distance_matrix_not_loaded(tsp_instance: TSPInstance):
    """Test get_distance_matrix returns None when not loaded."""
    result = tsp_instance.get_distance_matrix()
    assert result is None


def test_get_distance_matrix_loaded(monkeypatch, tsp_instance: TSPInstance):
    """Test get_distance_matrix returns valid matrix."""
    tsp_instance.has_loaded = True
    tsp_instance.distance_matrix = [[0, 1], [1, 0]]
    result = tsp_instance.get_distance_matrix()
    assert result == [[0, 1], [1, 0]]
