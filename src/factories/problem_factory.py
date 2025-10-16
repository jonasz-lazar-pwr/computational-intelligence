from typing import Any

from src.interfaces.factories_interfaces import IProblemFactory
from src.interfaces.problems_interfaces import IProblem
from src.problems.tsp.tsp_instance import TSPInstance
from src.problems.tsp.tsp_problem import TSPProblem


class ProblemFactory(IProblemFactory):
    """Factory for creating problem instances."""

    @staticmethod
    def build(problem_type: str, **kw: Any) -> IProblem:
        """Build and return a problem instance by type."""
        if problem_type == "tsp":
            instance = TSPInstance(**kw)
            return TSPProblem(instance)
        raise ValueError(f"Unknown problem type: {problem_type}")
