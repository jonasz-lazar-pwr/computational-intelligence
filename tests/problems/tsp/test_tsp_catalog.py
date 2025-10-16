import json

from pathlib import Path

import pytest

from src.problems.tsp.tsp_catalog import TSPCatalog
from src.problems.tsp.tsp_instance import TSPInstance


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Return temporary directory with valid and invalid TSP files."""
    good_tsp = tmp_path / "berlin52.tsp"
    bad_tsp = tmp_path / "invalid.tsp"
    not_tsp = tmp_path / "README.txt"
    good_tsp.write_text(
        """NAME: berlin52
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 10 10
2 20 10
3 20 20
EOF
""",
        encoding="utf-8",
    )
    bad_tsp.write_text("INVALID CONTENT", encoding="utf-8")
    not_tsp.write_text("Not a tsp file", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def optimal_json(tmp_path: Path) -> Path:
    """Return temporary optimal.json file."""
    data = {"berlin52": 7542}
    file_path = tmp_path / "optimal.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


@pytest.fixture()
def tsp_catalog(optimal_json: Path) -> TSPCatalog:
    """Provide initialized TSPCatalog instance."""
    return TSPCatalog(str(optimal_json))


def test_load_files_and_list_instances(tsp_catalog: TSPCatalog, data_dir: Path):
    """Test loading and listing instances."""
    tsp_catalog.load_files(str(data_dir))
    names = tsp_catalog.list_instances()
    assert isinstance(names, list)
    assert "berlin52" in names
    assert len(names) == 1


def test_get_file_by_name_success(tsp_catalog: TSPCatalog, data_dir: Path):
    """Test successful file retrieval by name."""
    tsp_catalog.load_files(str(data_dir))
    inst = tsp_catalog.get_file_by_name("berlin52")
    assert inst is not None
    assert inst.name == "berlin52"


def test_get_file_by_name_missing(tsp_catalog: TSPCatalog, data_dir: Path):
    """Test missing file by name."""
    tsp_catalog.load_files(str(data_dir))
    result = tsp_catalog.get_file_by_name("nonexistent")
    assert result is None


def test_get_file_by_name_error(monkeypatch, tsp_catalog: TSPCatalog):
    """Test iteration error when accessing .name."""

    class BrokenInstance(TSPInstance):
        def __init__(self, file_path: str, optimal_results_path: str) -> None:
            self.file_path = file_path
            self.optimal_results_path = optimal_results_path

        @property
        def name(self) -> str:
            raise RuntimeError("boom")

    tsp_catalog.instances = [BrokenInstance("fake.tsp", "fake.json")]
    result = tsp_catalog.get_file_by_name("whatever")
    assert result is None


def test_load_files_directory_missing(tsp_catalog: TSPCatalog):
    """Test missing directory error."""
    with pytest.raises(FileNotFoundError):
        tsp_catalog.load_files("nonexistent_dir")


def test_load_files_invalid_tsp(monkeypatch, tsp_catalog: TSPCatalog, data_dir: Path):
    """Test ValueError during TSP parsing."""

    def bad_load_metadata(_):
        raise ValueError("bad TSP")

    monkeypatch.setattr(TSPInstance, "load_metadata", bad_load_metadata)
    tsp_catalog.load_files(str(data_dir))
    assert len(tsp_catalog.instances) == 0


def test_load_files_unexpected_error(monkeypatch, tsp_catalog: TSPCatalog, data_dir: Path):
    """Test unexpected error during instance creation."""

    def bad_init(*_, **__):
        raise RuntimeError("explode")

    monkeypatch.setattr("src.problems.tsp.tsp_catalog.TSPInstance", bad_init)
    tsp_catalog.load_files(str(data_dir))
    assert len(tsp_catalog.instances) == 0


def test_clear_files(tsp_catalog: TSPCatalog, data_dir: Path):
    """Test clearing loaded files."""
    tsp_catalog.load_files(str(data_dir))
    assert len(tsp_catalog.instances) > 0
    tsp_catalog.clear_files()
    assert len(tsp_catalog.instances) == 0


def test_load_files_directory_vanished(monkeypatch, tsp_catalog: TSPCatalog, tmp_path: Path):
    """Test vanished directory during iteration."""
    ghost_tsp = tmp_path / "ghost.tsp"
    ghost_tsp.write_text(
        """NAME: ghost
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 1
EOF
""",
        encoding="utf-8",
    )

    def bad_iterdir(_self):
        yield ghost_tsp
        raise FileNotFoundError("directory vanished")

    monkeypatch.setattr(Path, "iterdir", bad_iterdir)
    with pytest.raises(FileNotFoundError, match="directory vanished"):
        tsp_catalog.load_files(str(tmp_path))


def test_summary_with_and_without_instances(tsp_catalog: TSPCatalog, data_dir: Path):
    """Test summary output with and without loaded instances."""
    tsp_catalog.summary()
    tsp_catalog.load_files(str(data_dir))
    tsp_catalog.summary()
