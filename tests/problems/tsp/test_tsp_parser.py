from pathlib import Path

import pytest

from src.problems.tsp.tsp_parser import TSPParser


@pytest.fixture()
def parser() -> TSPParser:
    """Return TSPParser instance."""
    return TSPParser()


@pytest.fixture()
def tmp_tsp(tmp_path: Path) -> Path:
    """Return valid EUC_2D TSP file."""
    content = """NAME: test
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10 10
2 20 10
3 20 20
EOF
"""
    file_path = tmp_path / "test_euc2d.tsp"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_validate_file_success(parser: TSPParser, tmp_tsp: Path):
    """Test successful validation of TSP file."""
    parser.validate_file(str(tmp_tsp))
    assert parser.edge_weight_type == "EUC_2D"
    assert len(parser.coordinates) == 3


def test_validate_file_missing_file(parser: TSPParser):
    """Test missing TSP file raises error."""
    with pytest.raises(FileNotFoundError):
        parser.validate_file("nonexistent_file.tsp")


def test_validate_file_empty(parser: TSPParser, tmp_path: Path):
    """Test empty TSP file raises ValueError."""
    f = tmp_path / "empty.tsp"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        parser.validate_file(str(f))


def test_validate_file_missing_field(parser: TSPParser, tmp_path: Path):
    """Test missing DIMENSION field raises ValueError."""
    f = tmp_path / "invalid.tsp"
    f.write_text("NAME: test\nTYPE: TSP\nEDGE_WEIGHT_TYPE: EUC_2D\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Field DIMENSION not found"):
        parser.validate_file(str(f))


def test_validate_file_invalid_edge_type(parser: TSPParser, tmp_path: Path):
    """Test unsupported EDGE_WEIGHT_TYPE raises ValueError."""
    f = tmp_path / "invalid_edge.tsp"
    f.write_text(
        "NAME: test\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: UNKNOWN\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_TYPE"):
        parser.validate_file(str(f))


def test_validate_file_explicit(parser: TSPParser, tmp_path: Path):
    """Test EXPLICIT edge weight format parsing."""
    f = tmp_path / "explicit.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1
1 0
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    assert parser.edge_weight_type == "EXPLICIT"
    assert len(parser.distance_matrix) == 2


def test_load_coordinates_and_display(parser: TSPParser, tmp_tsp: Path):
    """Test coordinate and display data loading."""
    parser.validate_file(str(tmp_tsp))
    coords = parser.coordinates
    display_coords = parser.load_display_coordinates()
    assert len(coords) == 3
    assert display_coords == []


def test_load_coordinates_malformed(parser: TSPParser, tmp_path: Path):
    """Test malformed coordinate data is ignored."""
    f = tmp_path / "malformed_coords.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 x y
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    assert len(parser.coordinates) == 0


@pytest.mark.parametrize("edge_type", ["EUC_2D", "CEIL_2D", "ATT", "GEO"])
def test_generate_distance_matrix_all(parser: TSPParser, tmp_tsp: Path, edge_type):
    """Test distance matrix generation for multiple edge types."""
    content = Path(tmp_tsp).read_text().replace("EUC_2D", edge_type)
    Path(tmp_tsp).write_text(content, encoding="utf-8")
    parser.validate_file(str(tmp_tsp))
    parser.generate_distance_matrix()
    matrix = parser.get_distance_matrix()
    assert len(matrix) == 3
    for i in range(3):
        assert matrix[i][i] == 0


def test_generate_distance_matrix_explicit(parser: TSPParser, tmp_path: Path):
    """Test distance matrix generation for EXPLICIT format."""
    f = tmp_path / "explicit.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1
1 0
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    parser.generate_distance_matrix()
    assert len(parser.distance_matrix) == 2


def test_generate_distance_matrix_invalid_type(parser: TSPParser):
    """Test unsupported edge type raises ValueError."""
    parser.edge_weight_type = "INVALID"
    parser.coordinates = [(0, 0), (1, 1)]
    parser.file_path = Path("fake.tsp")
    with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_TYPE"):
        parser.generate_distance_matrix()


@pytest.mark.parametrize(
    "fmt", ["FULL_MATRIX", "LOWER_ROW", "LOWER_DIAG_ROW", "UPPER_ROW", "UPPER_DIAG_ROW"]
)
def test_load_explicit_formats(parser: TSPParser, tmp_path: Path, fmt: str):
    """Test supported explicit matrix formats."""
    f = tmp_path / f"{fmt.lower()}.tsp"
    f.write_text(
        f"""NAME: test
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: {fmt}
EDGE_WEIGHT_SECTION
0 1 2 1 0 1 2 1 0
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    assert len(parser.distance_matrix) == 3
    assert all(len(r) == 3 for r in parser.distance_matrix)


def test_load_explicit_unsupported(parser: TSPParser, tmp_path: Path):
    """Test unsupported EDGE_WEIGHT_FORMAT raises ValueError."""
    f = tmp_path / "unsupported.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: UNKNOWN
EDGE_WEIGHT_SECTION
0 1 1 0
EOF
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_FORMAT"):
        parser.validate_file(str(f))


def test_load_explicit_no_values(parser: TSPParser, tmp_path: Path):
    """Test missing numeric values raises ValueError."""
    f = tmp_path / "novalues.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
EOF
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No numeric values"):
        parser.validate_file(str(f))


def test_load_explicit_malformed_line(parser: TSPParser, tmp_path: Path):
    """Test malformed EDGE_WEIGHT_SECTION raises ValueError."""
    f = tmp_path / "malformed_values.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
a b
EOF
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        parser.validate_file(str(f))


def test_convert_to_int_and_get_field_value(parser: TSPParser):
    """Test helper methods for conversion and field lookup."""
    parser.distance_matrix = [[1, 2], [3, 4]]
    parser._convert_to_int()
    assert parser.distance_matrix == [[1, 2], [3, 4]]
    parser.file_path = Path("dummy.tsp")
    parser.content = "FIELD1: value\nFIELD2: something"
    assert parser.get_field_value("FIELD1") == "value"
    with pytest.raises(ValueError, match="Field NOT_FOUND not found"):
        parser.get_field_value("NOT_FOUND")


def test_get_distance_matrix_warns(parser: TSPParser):
    """Test warning on empty distance matrix."""
    parser.distance_matrix = []
    parser.file_path = Path("dummy.tsp")
    result = parser.get_distance_matrix()
    assert result == []


def test_geo_coordinates_parsing(parser: TSPParser, tmp_path: Path):
    """Test GEO coordinates parsing in degrees/minutes."""
    f = tmp_path / "geo.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: GEO
NODE_COORD_SECTION
1 52.22 21.01
2 50.06 19.94
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    parser.generate_distance_matrix()
    matrix = parser.get_distance_matrix()
    assert len(matrix) == 2
    assert matrix[0][1] > 0


def test_display_data_section_invalid(parser: TSPParser, tmp_path: Path):
    """Test invalid DISPLAY_DATA_SECTION returns empty list."""
    f = tmp_path / "display_invalid.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10 10
2 20 20
DISPLAY_DATA_SECTION
1 a b
EOF
""",
        encoding="utf-8",
    )
    parser.validate_file(str(f))
    display = parser.load_display_coordinates()
    assert display == []


def test_unsupported_format_direct(parser: TSPParser, tmp_path: Path):
    """Test _unsupported_format raises ValueError."""
    f = tmp_path / "unsupported_fmt.tsp"
    f.write_text(
        """NAME: test
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: TRIANGULAR_RANDOM
EDGE_WEIGHT_SECTION
0 1
1 0
EOF
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_FORMAT"):
        parser.validate_file(str(f))
