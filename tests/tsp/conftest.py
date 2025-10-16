import json

from pathlib import Path

import pytest

from src.interfaces.i_tsp_catalog import ITSPCatalog
from src.interfaces.i_tsp_instance import ITSPInstance
from src.interfaces.i_tsp_parser import ITSPParser
from src.problems.tsp.tsp_catalog import TSPCatalog
from src.problems.tsp.tsp_instance import TSPInstance
from src.problems.tsp.tsp_parser import TSPParser


@pytest.fixture()
def optimal_results_path(tmp_path: Path) -> str:
    """Return temporary optimal.json file."""
    data = {"berlin52": 7542}
    file_path = tmp_path / "optimal.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return str(file_path)


@pytest.fixture()
def parser() -> ITSPParser:
    """Return TSPLIB parser instance."""
    return TSPParser()


@pytest.fixture()
def tsp_instance(tmp_path: Path, optimal_results_path: str) -> ITSPInstance:
    """Return TSPInstance with temporary TSP file."""
    tsp_file = tmp_path / "berlin52.tsp"
    tsp_file.write_text(
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
    return TSPInstance(str(tsp_file), optimal_results_path)


@pytest.fixture()
def tsp_catalog(optimal_results_path: str) -> ITSPCatalog:
    """Return TSPCatalog using the same optimal.json."""
    return TSPCatalog(optimal_results_path)
