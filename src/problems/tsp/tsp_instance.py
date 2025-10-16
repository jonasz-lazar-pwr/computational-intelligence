import json

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.core.logger import get_logger
from src.interfaces.tsp_interfaces import ITSPInstance, ITSPParser
from src.problems.tsp.tsp_parser import TSPParser

logger = get_logger(__name__)


class TSPInstance(ITSPInstance):
    """Represent a single TSP instance including metadata, coordinates, and optimal result."""

    def __init__(self, file_path: str, optimal_results_path: str) -> None:
        """Initialize TSPInstance with paths and default attributes."""
        self.file_path: Path = Path(file_path)
        self.name: Optional[str] = None
        self.type: Optional[str] = None
        self.dimension: Optional[int] = None
        self.edge_weight_type: Optional[str] = None
        self.edge_weight_format: Optional[str] = None
        self.coordinates: List[Tuple[float, float]] = []
        self.display_coordinates: List[Tuple[float, float]] = []
        self.distance_matrix: List[List[int]] = []
        self.has_loaded: bool = False
        self.optimal_result: Optional[int] = None
        self.optimal_results_path: Path = Path(optimal_results_path)
        self.parser: ITSPParser = TSPParser()

    def load_metadata(self) -> None:
        """Load metadata and coordinates from a TSPLIB file."""
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            logger.debug(f"Loading metadata for: {self.file_path.name}")
            self.parser.validate_file(str(self.file_path))
            self.name = self.parser.get_field_value("NAME")
            self.type = self.parser.get_field_value("TYPE")
            self.dimension = int(self.parser.get_field_value("DIMENSION"))
            self.edge_weight_type = self.parser.get_field_value("EDGE_WEIGHT_TYPE")
            self.edge_weight_format = self.parser.get_field_value(
                "EDGE_WEIGHT_FORMAT", optional=True
            )
            logger.info(
                f"Parsed metadata: {self.name} ({self.dimension} cities, type={self.edge_weight_type})"
            )
            self._load_optimal_result()
            if self.edge_weight_type != "EXPLICIT":
                self.coordinates = self.parser.coordinates
            if self.edge_weight_type == "EXPLICIT":
                self.load_distance_matrix()
            display_data_type = self.parser.get_field_value("DISPLAY_DATA_TYPE", optional=True)
            if display_data_type == "TWOD_DISPLAY":
                self.load_display_coordinates()
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Invalid or missing TSP file: {self.file_path} — {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while loading metadata for {self.file_path}: {e}", exc_info=True
            )
            raise

    def _load_optimal_result(self) -> None:
        """Load known optimal result from JSON file."""
        try:
            if not self.optimal_results_path.exists():
                raise FileNotFoundError(
                    f"Optimal results JSON not found: {self.optimal_results_path}"
                )
            with self.optimal_results_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid format in {self.optimal_results_path}. Expected dict.")
            key = self.file_path.name.replace(".tsp", "")
            self.optimal_result = data.get(key)
            if self.optimal_result is not None:
                logger.debug(f"Loaded optimal result for {key}: {self.optimal_result}")
            else:
                logger.info(f"No optimal result found for {key} in JSON file.")
        except FileNotFoundError as e:
            logger.warning(str(e))
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON in {self.optimal_results_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading optimal results: {e}", exc_info=True)
            self.optimal_result = None

    def load_display_coordinates(self) -> None:
        """Load display coordinates from DISPLAY_DATA_SECTION."""
        try:
            self.display_coordinates = self.parser.load_display_coordinates()
            logger.debug(
                f"Loaded {len(self.display_coordinates)} display coordinates for {self.name}"
            )
        except Exception as e:
            logger.warning(f"Failed to load display coordinates for {self.file_path}: {e}")

    def load_distance_matrix(self) -> None:
        """Generate or retrieve the distance matrix."""
        try:
            if not self.has_loaded:
                self.parser.generate_distance_matrix()
                self.distance_matrix = self.parser.get_distance_matrix()
                self.has_loaded = True
                logger.debug(f"Distance matrix generated for {self.name or self.file_path.name}")
            else:
                logger.info(
                    f"Distance matrix for {self.name or self.file_path.name} already loaded."
                )
        except Exception as e:
            logger.error(
                f"Failed to generate distance matrix for {self.file_path}: {e}", exc_info=True
            )
            raise

    def get_distance_matrix(self) -> Optional[List[List[int]]]:
        """Return distance matrix if loaded, otherwise None."""
        if self.has_loaded:
            return self.distance_matrix
        logger.warning(f"Distance matrix not loaded yet for {self.name or self.file_path.name}.")
        return None

    def to_dict(self) -> Dict:
        """Return instance data as a serializable dictionary."""
        logger.debug(f"Serializing TSP instance to dict: {self.name}")
        return {
            "file_path": str(self.file_path),
            "name": self.name,
            "type": self.type,
            "dimension": self.dimension,
            "edge_weight_type": self.edge_weight_type,
            "edge_weight_format": self.edge_weight_format,
            "optimal_length": self.optimal_result,
            "coordinates": self.coordinates,
            "display_coordinates": self.display_coordinates,
            "distance_matrix": self.distance_matrix if self.has_loaded else None,
            "has_loaded": self.has_loaded,
            "optimal_results_path": str(self.optimal_results_path),
        }
