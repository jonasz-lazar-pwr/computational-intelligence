from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


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


class ITSPInstance(ABC):
    """Interface for a single TSP problem instance."""

    file_path: str
    name: Optional[str]
    dimension: Optional[int]
    edge_weight_type: Optional[str]
    has_loaded: bool
    optimal_result: Optional[int]

    @abstractmethod
    def load_metadata(self) -> None:
        """Load instance metadata from source file."""
        pass

    @abstractmethod
    def load_distance_matrix(self) -> None:
        """Generate or load the distance matrix."""
        pass

    @abstractmethod
    def load_display_coordinates(self) -> None:
        """Load display coordinates if available."""
        pass

    @abstractmethod
    def get_distance_matrix(self) -> Optional[List[List[int]]]:
        """Return the distance matrix if loaded."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """Export instance data as a dictionary."""
        pass


class ITSPCatalog(ABC):
    """Interface for managing multiple TSP instances."""

    @abstractmethod
    def clear_files(self) -> None:
        """Clear all loaded TSP instances."""
        pass

    @abstractmethod
    def load_files(self, directory_path: str) -> None:
        """Load .tsp instances from a directory."""
        pass

    @abstractmethod
    def get_file_by_name(self, name: str) -> Optional[ITSPInstance]:
        """Return a TSP instance by name."""
        pass

    @abstractmethod
    def list_instances(self) -> List[str]:
        """List names of loaded instances."""
        pass

    @abstractmethod
    def summary(self) -> None:
        """Log a summary of loaded instances."""
        pass
