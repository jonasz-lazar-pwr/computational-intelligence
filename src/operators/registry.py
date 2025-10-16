from typing import Any

from src.core.logger import get_logger
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection
from src.operators.crossover.ox import OrderCrossover
from src.operators.mutation.swap import SwapMutation
from src.operators.selection.tournament import TournamentSelection

logger = get_logger(__name__)


def build_selection(name: str, **kw: Any) -> ISelection:
    """Return a selection operator by name."""
    if name == "tournament":
        return TournamentSelection(**kw)
    raise ValueError(f"Unknown selection: {name}")


def build_crossover(name: str, **kw: Any) -> ICrossover:
    """Return a crossover operator by name."""
    if name == "ox":
        return OrderCrossover(**kw)
    raise ValueError(f"Unknown crossover: {name}")


def build_mutation(name: str, **kw: Any) -> IMutation:
    """Return a mutation operator by name."""
    if name == "swap":
        return SwapMutation(**kw)
    raise ValueError(f"Unknown mutation: {name}")
