from typing import Any, ClassVar

from src.algorithms.acs_algorithm import ACSAlgorithm
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.factories.operator_factory import OperatorFactory
from src.interfaces.factories_interfaces import IAlgorithmFactory


class AlgorithmFactory(IAlgorithmFactory):
    """Create algorithm instances for GA and ACS."""

    _REGISTRY: ClassVar[dict[str, Any]] = {
        "ga": GeneticAlgorithm,
        "acs": ACSAlgorithm,
    }

    @classmethod
    def build(cls, name: str, **config: Any):
        """Build an algorithm instance by name."""
        try:
            algorithm_cls = cls._REGISTRY[name]
        except KeyError as err:
            raise ValueError(f"Unknown algorithm: {name}") from err

        if name == "ga":
            return cls._build_ga(algorithm_cls, config)
        if name == "acs":
            return cls._build_acs(algorithm_cls, config)

        raise ValueError(f"Unsupported algorithm type: {name}")

    @staticmethod
    def _build_ga(algorithm_cls, config: dict):
        """Construct a GeneticAlgorithm instance."""
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
            seed=config.get("seed"),
        )

    @staticmethod
    def _build_acs(algorithm_cls, config: dict):
        """Construct an ACSAlgorithm instance."""
        return algorithm_cls(
            problem=config["problem"],
            num_ants=config["num_ants"],
            alpha=config["alpha"],
            beta=config["beta"],
            rho=config["rho"],
            phi=config["phi"],
            q0=config["q0"],
            max_time=config["max_time"],
            seed=config.get("seed"),
        )
