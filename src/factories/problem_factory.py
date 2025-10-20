from copy import deepcopy
from typing import Any, ClassVar

from src.interfaces.factories_interfaces import IProblemFactory
from src.interfaces.problems_interfaces import IProblem
from src.problems.tsp.tsp_instance import TSPInstance
from src.problems.tsp.tsp_problem import TSPProblem


class ProblemFactory(IProblemFactory):
    """Creates problem instances such as TSP."""

    _REGISTRY: ClassVar[dict[str, Any]] = {"tsp": (TSPInstance, TSPProblem)}

    @classmethod
    def build(cls, name: str, **config: Any) -> IProblem:
        """Construct and return a problem instance based on name and configuration."""
        try:
            instance_cls, problem_cls = cls._REGISTRY[name]
        except KeyError as err:
            raise ValueError(f"Unknown problem: {name}") from err

        safe_config = deepcopy(config)

        instance = instance_cls(**safe_config)
        return problem_cls(instance)
