from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ITSPInstance(ABC):
    """Interface for a single TSP problem instance."""

    file_path: str
    name: Optional[str]
    dimension: Optional[int]
    edge_weight_type: Optional[str]
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
