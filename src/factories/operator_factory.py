from typing import Any, ClassVar

from src.interfaces.factories_interfaces import IOperatorFactory
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection, ISuccession
from src.operators.crossover.cx import CycleCrossover
from src.operators.crossover.ox import OrderCrossover
from src.operators.crossover.pmx import PartiallyMappedCrossover
from src.operators.mutation.insert import InsertMutation
from src.operators.mutation.swap import SwapMutation
from src.operators.selection.rank import RankSelection
from src.operators.selection.roulette import RouletteSelection
from src.operators.selection.tournament import TournamentSelection
from src.operators.succession.elitist import ElitistSuccession
from src.operators.succession.steady_state import SteadyStateSuccession


class OperatorFactory(IOperatorFactory):
    """Creates operator instances for selection, crossover, mutation, and succession."""

    _REGISTRY: ClassVar[dict[str, dict[str, Any]]] = {
        "selection": {
            "tournament": TournamentSelection,
            "roulette": RouletteSelection,
            "rank": RankSelection,
        },
        "crossover": {
            "ox": OrderCrossover,
            "cx": CycleCrossover,
            "pmx": PartiallyMappedCrossover,
        },
        "mutation": {
            "insert": InsertMutation,
            "swap": SwapMutation,
        },
        "succession": {
            "elitist": ElitistSuccession,
            "steady_state": SteadyStateSuccession,
        },
    }

    @classmethod
    def get_operator(
        cls, category: str, name: str, **config: Any
    ) -> ISelection | ICrossover | IMutation | ISuccession:
        """Return an operator instance based on category and name."""
        try:
            return cls._REGISTRY[category][name](**config)
        except KeyError as err:
            raise ValueError(f"Unknown operator: {category}/{name}") from err
