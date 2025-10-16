import math

from pathlib import Path
from typing import List, Optional, Tuple

from src.core.logger import get_logger
from src.interfaces.tsp_interfaces import ITSPParser

logger = get_logger(__name__)
COORD_PARTS_COUNT = 3


class TSPParser(ITSPParser):
    """Parse TSPLIB-formatted .tsp files supporting coordinates and explicit matrices."""

    def __init__(self, file_path: Path | None = None) -> None:
        """Initialize parser with optional file path."""
        self.file_path: Optional[Path] = file_path
        self.content: Optional[str] = None
        self.coordinates: List[Tuple[float, float]] = []
        self.distance_matrix: List[List[int]] = []
        self.edge_weight_type: Optional[str] = None
        self.edge_weight_format: Optional[str] = None
        logger.debug(f"Initialized TSPParser for {self.file_path}")

    def validate_file(self, file_path: str) -> None:
        """Read and validate TSPLIB file structure."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            logger.error(f"TSP file not found: {file_path}")
            raise FileNotFoundError(f"TSP file not found: {file_path}")
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                self.content = f.read()
            if not self.content.strip():
                raise ValueError("TSP file is empty.")
            required_fields = ["NAME", "TYPE", "DIMENSION", "EDGE_WEIGHT_TYPE"]
            supported_types = {"EXPLICIT", "EUC_2D", "CEIL_2D", "ATT", "GEO"}
            for field in required_fields:
                field_value = self.get_field_value(field)
                if not field_value:
                    raise ValueError(f"Missing required field: {field}")
                if field == "DIMENSION" and (not field_value.isdigit() or int(field_value) <= 0):
                    raise ValueError(f"Invalid DIMENSION value: {field_value}")
            self.edge_weight_type = self.get_field_value("EDGE_WEIGHT_TYPE")
            self.edge_weight_format = self.get_field_value("EDGE_WEIGHT_FORMAT", optional=True)
            if self.edge_weight_type not in supported_types:
                raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {self.edge_weight_type}")
            if self.edge_weight_type == "EXPLICIT":
                self._load_explicit_weights()
            elif "NODE_COORD_SECTION" in self.content:
                self._load_coordinates()
            else:
                raise ValueError("Unrecognized TSP structure.")
            logger.info(f"Validated and parsed TSP file: {self.file_path.name}")
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while validating {self.file_path.name}: {e}", exc_info=True
            )
            raise

    def _load_coordinates(self) -> None:
        """Load NODE_COORD_SECTION coordinates."""
        self.coordinates.clear()
        if not self.content:
            return
        try:
            in_section = False
            for line in self.content.splitlines():
                if "NODE_COORD_SECTION" in line:
                    in_section = True
                    continue
                if in_section:
                    if line.strip() == "EOF":
                        break
                    parts = line.split()
                    if len(parts) >= COORD_PARTS_COUNT:
                        try:
                            _, x, y = parts[:3]
                            self.coordinates.append((float(x), float(y)))
                        except ValueError as e:
                            logger.warning(f"Skipping malformed coordinate line: {line} ({e})")
                            continue
        except Exception as e:
            logger.error(f"Error loading coordinates: {e}", exc_info=True)
            raise

    def load_display_coordinates(self) -> List[Tuple[float, float]]:
        """Load DISPLAY_DATA_SECTION coordinates."""
        display_coordinates: List[Tuple[float, float]] = []
        if not self.content:
            return display_coordinates
        try:
            in_section = False
            for line in self.content.splitlines():
                if "DISPLAY_DATA_SECTION" in line:
                    in_section = True
                    continue
                if in_section:
                    if line.strip() == "EOF":
                        break
                    parts = line.split()
                    if len(parts) == COORD_PARTS_COUNT:
                        try:
                            _, x, y = parts
                            display_coordinates.append((float(x), float(y)))
                        except ValueError as e:
                            logger.warning(
                                f"Skipping malformed display coordinate line: {line} ({e})"
                            )
                            continue
        except Exception as e:
            logger.error(f"Error loading display coordinates: {e}", exc_info=True)
            raise
        return display_coordinates

    def generate_distance_matrix(self) -> None:
        """Generate distance matrix from coordinates."""
        if not self.coordinates and self.edge_weight_type != "EXPLICIT":
            return
        try:
            match self.edge_weight_type:
                case "EUC_2D":
                    self._calculate_euclidean()
                case "CEIL_2D":
                    self._calculate_ceil_euclidean()
                case "ATT":
                    self._calculate_att()
                case "GEO":
                    self._calculate_geographical()
                case "EXPLICIT":
                    return
                case _:
                    raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {self.edge_weight_type}")
        except Exception as e:
            logger.error(f"Error generating distance matrix: {e}", exc_info=True)
            raise

    def _calculate_euclidean(self) -> None:
        """Calculate EUC_2D distances."""
        n = len(self.coordinates)
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]
                d = int(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 0.5)
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = d

    def _calculate_ceil_euclidean(self) -> None:
        """Calculate CEIL_2D distances."""
        n = len(self.coordinates)
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self.coordinates[i]
                x2, y2 = self.coordinates[j]
                d = math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = int(d)

    def _calculate_att(self) -> None:
        """Calculate ATT pseudo-Euclidean distances."""
        n = len(self.coordinates)
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                xd = self.coordinates[i][0] - self.coordinates[j][0]
                yd = self.coordinates[i][1] - self.coordinates[j][1]
                rij = math.sqrt((xd**2 + yd**2) / 10.0)
                tij = int(rij + 0.5)
                dij = tij + 1 if tij < rij else tij
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = dij

    def _calculate_geographical(self) -> None:
        """Calculate GEO distances in degrees."""

        def to_radians(deg_min: float) -> float:
            deg = int(deg_min)
            min_ = deg_min - deg
            return math.pi * (deg + 5.0 * min_ / 3.0) / 180.0

        radius = 6378.388
        n = len(self.coordinates)
        lat = [to_radians(c[0]) for c in self.coordinates]
        lon = [to_radians(c[1]) for c in self.coordinates]
        self.distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                q1 = math.cos(lon[i] - lon[j])
                q2 = math.cos(lat[i] - lat[j])
                q3 = math.cos(lat[i] + lat[j])
                d = int(radius * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = d

    def _load_explicit_weights(self) -> None:
        """Parse EDGE_WEIGHT_SECTION for explicit formats."""
        self.distance_matrix.clear()
        if not self.content:
            return
        try:
            values: List[int] = []
            in_section = False
            dimension = int(self.get_field_value("DIMENSION"))
            for line in self.content.splitlines():
                if any(s in line for s in ["DISPLAY_DATA_SECTION", "EOF", "NODE_COORD_SECTION"]):
                    break
                if "EDGE_WEIGHT_SECTION" in line:
                    in_section = True
                    continue
                if in_section:
                    values.extend(map(int, line.split()))
            if not values:
                raise ValueError("No numeric values found in EDGE_WEIGHT_SECTION.")
            load_methods = {
                "FULL_MATRIX": lambda: self._load_full_matrix(values, dimension),
                "LOWER_DIAG_ROW": lambda: self._load_triangular(values, dimension, True, True),
                "LOWER_ROW": lambda: self._load_triangular(values, dimension, True, False),
                "UPPER_DIAG_ROW": lambda: self._load_triangular(values, dimension, False, True),
                "UPPER_ROW": lambda: self._load_triangular(values, dimension, False, False),
            }
            if self.edge_weight_format not in load_methods:
                self._unsupported_format()
            else:
                load_methods[self.edge_weight_format]()
        except Exception as e:
            logger.error(f"Error parsing explicit weights: {e}", exc_info=True)
            raise

    def _load_full_matrix(self, values: List[int], n: int) -> None:
        """Load FULL_MATRIX distance matrix."""
        self.distance_matrix = [values[i * n : (i + 1) * n] for i in range(n)]
        self._convert_to_int()

    def _load_triangular(self, values: List[int], n: int, lower: bool, diag: bool) -> None:
        """Load LOWER/UPPER (DIAG) ROW matrix formats."""
        self.distance_matrix = [[0] * n for _ in range(n)]
        k = 0
        for i in range(n):
            if lower:
                for j in range(i + 1 if diag else i):
                    if k < len(values):
                        v = values[k]
                        self.distance_matrix[i][j] = self.distance_matrix[j][i] = v
                        k += 1
            else:
                for j in range(i if diag else i + 1, n):
                    if k < len(values):
                        v = values[k]
                        self.distance_matrix[i][j] = self.distance_matrix[j][i] = v
                        k += 1
        self._convert_to_int()

    def _convert_to_int(self) -> None:
        """Ensure all matrix values are integers."""
        self.distance_matrix = [[int(v) for v in row] for row in self.distance_matrix]

    def _unsupported_format(self) -> None:
        """Raise error for unsupported EDGE_WEIGHT_FORMAT."""
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {self.edge_weight_format}")

    def get_field_value(self, field: str, optional: bool = False) -> Optional[str]:
        """Extract field value from TSPLIB content."""
        for line in self.content.splitlines():
            if line.startswith(field):
                parts = line.split(":")
                if len(parts) > 1:
                    return parts[1].strip()
        if not optional:
            raise ValueError(f"Field {field} not found in {self.file_path.name}")
        return None

    def get_distance_matrix(self) -> List[List[int]]:
        """Return generated or parsed distance matrix."""
        if not self.distance_matrix:
            logger.warning(f"No distance matrix available for {self.file_path.name}")
        return self.distance_matrix
