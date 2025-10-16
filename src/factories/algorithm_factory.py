from typing import Any

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.factories_interfaces import IAlgorithmFactory


class AlgorithmFactory(IAlgorithmFactory):
    """Factory for building optimization algorithms."""

    @staticmethod
    def build(algorithm_type: str, **kw: Any) -> IAlgorithm:
        """Build and return an algorithm instance by name."""
        if algorithm_type == "ga":
            return GeneticAlgorithm(**kw)
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
