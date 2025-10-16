from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class ITSPParser(ABC):
    """Interface for TSPLIB parsers."""

    coordinates: List[Tuple[float, float]]
    edge_weight_type: Optional[str]

    @abstractmethod
    def validate_file(self, file_path: str) -> None:
        """Validate and read a TSPLIB file."""
        pass

    @abstractmethod
    def generate_distance_matrix(self) -> None:
        """Generate a distance matrix from coordinates or explicit data."""
        pass

    @abstractmethod
    def load_display_coordinates(self) -> List[Tuple[float, float]]:
        """Load DISPLAY_DATA_SECTION coordinates."""
        pass

    @abstractmethod
    def get_field_value(self, field: str, optional: bool = False) -> Optional[str]:
        """Return a metadata field value."""
        pass

    @abstractmethod
    def get_distance_matrix(self) -> List[List[int]]:
        """Return the generated or parsed distance matrix."""
        pass
