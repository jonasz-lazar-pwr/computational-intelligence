from abc import ABC, abstractmethod
from typing import Any, Dict


class IAlgorithm(ABC):
    """Abstract interface for all optimization algorithms."""

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the algorithm and return results."""
        pass
