from typing import Any

from src.interfaces.factories_interfaces import IOperatorFactory
from src.interfaces.operators_interfaces import ICrossover, IMutation, ISelection
from src.operators.crossover.ox import OrderCrossover
from src.operators.mutation.insert import InsertMutation
from src.operators.selection.tournament import TournamentSelection


class OperatorFactory(IOperatorFactory):
    """Factory for building selection, crossover, and mutation operators."""

    @staticmethod
    def get_operator(category: str, name: str, **kw: Any) -> ISelection | ICrossover | IMutation:
        """Return an operator instance based on category and name."""
        if category == "selection" and name == "tournament":
            return TournamentSelection(**kw)
        if category == "crossover" and name == "ox":
            return OrderCrossover(**kw)
        if category == "mutation" and name == "insert":
            return InsertMutation(**kw)
        raise ValueError(f"Unknown operator: {category}/{name}")
