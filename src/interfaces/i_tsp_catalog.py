from abc import ABC, abstractmethod
from typing import List, Optional

from src.interfaces.i_tsp_instance import ITSPInstance


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
