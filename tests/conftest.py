from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def base_dir() -> Path:
    """Return base project directory."""
    return Path.cwd()


@pytest.fixture(scope="session")
def data_dir(base_dir: Path) -> Path:
    """Return TSPLIB dataset directory."""
    return base_dir / "data" / "tsplib"


@pytest.fixture(scope="session")
def optimal_results_path(base_dir: Path) -> Path:
    """Return path to optimal TSP results JSON."""
    return base_dir / "data" / "optimal_results.json"


@pytest.fixture
def tsp_file_path(data_dir: Path) -> Path:
    """Return sample TSP file path."""
    return data_dir / "berlin52.tsp"
