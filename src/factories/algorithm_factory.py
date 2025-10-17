from typing import Any, ClassVar

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.factories.operator_factory import OperatorFactory
from src.interfaces.factories_interfaces import IAlgorithmFactory


class AlgorithmFactory(IAlgorithmFactory):
    """Creates algorithm instances based on name and configuration."""

    _REGISTRY: ClassVar[dict[str, Any]] = {"ga": GeneticAlgorithm}

    @classmethod
    def build(cls, name: str, **config: Any):
        """Construct and return an algorithm instance using provided configuration."""
        try:
            algorithm_cls = cls._REGISTRY[name]
        except KeyError as err:
            raise ValueError(f"Unknown algorithm: {name}") from err

        operator_factory = OperatorFactory()
        selection = operator_factory.get_operator("selection", **config["selection_config"])
        crossover = operator_factory.get_operator("crossover", **config["crossover_config"])
        mutation = operator_factory.get_operator("mutation", **config["mutation_config"])
        succession = operator_factory.get_operator("succession", **config["succession_config"])

        return algorithm_cls(
            problem=config["problem"],
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            succession=succession,
            population_size=config["population_size"],
            crossover_rate=config["crossover_rate"],
            mutation_rate=config["mutation_rate"],
            max_time=config["max_time"],
        )
