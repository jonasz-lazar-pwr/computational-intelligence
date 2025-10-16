from abc import ABC, abstractmethod
from typing import Any, Tuple

from src.core.logger import get_logger
from src.interfaces.algorithms_interfaces import IAlgorithm
from src.interfaces.problems_interfaces import IProblem

logger = get_logger(__name__)


class BaseAlgorithm(IAlgorithm, ABC):
    """Abstract base class implementing the core algorithm lifecycle."""

    def __init__(self, problem: IProblem, config: dict[str, Any]) -> None:
        """Initialize algorithm with injected problem and configuration."""
        super().__init__(problem, config)
        self.population: list[Any] = []
        self.best_solution: Tuple[list[int], float] | None = None
        self.current_generation: int = 0
        self.max_generations: int = config.get("generations", 100)
        self.history: list[float] = []

    # ------------------------------------------------------------------
    # Template method (defines the algorithm lifecycle)
    # ------------------------------------------------------------------
    def run(self) -> Tuple[list[int], float]:
        """Template method controlling the algorithm flow."""
        logger.info(f"Starting algorithm: {self.__class__.__name__}")
        self.reset()
        self.initialize_population()

        for gen in range(self.max_generations):
            self.current_generation = gen
            logger.debug(f"Generation {gen + 1}/{self.max_generations}")
            self.evolve()
            self.evaluate_generation()
            self.log_progress()

        logger.info(f"Finished {self.__class__.__name__} after {self.max_generations} generations")
        return self.get_best_solution()

    # ------------------------------------------------------------------
    # Abstract methods for subclass override
    # ------------------------------------------------------------------
    @abstractmethod
    def initialize_population(self) -> None:
        """Initialize starting population."""
        pass

    @abstractmethod
    def evolve(self) -> None:
        """Execute a single iteration (selection, crossover, mutation, etc.)."""
        pass

    @abstractmethod
    def get_best_solution(self) -> Tuple[list[int], float]:
        """Return current best solution."""
        pass

    # ------------------------------------------------------------------
    # Common helpers (shared logic)
    # ------------------------------------------------------------------
    def evaluate_generation(self) -> None:
        """Evaluate all individuals and update the best solution."""
        best = min(self.population, key=lambda s: self.problem.evaluate(s))
        fitness = self.problem.evaluate(best)

        if self.best_solution is None or fitness < self.best_solution[1]:
            self.best_solution = (best, fitness)

        self.history.append(fitness)
        logger.debug(f"Best fitness this gen: {fitness:.3f}")

    def log_progress(self) -> None:
        """Log current best result and average metrics."""
        if not self.history:
            return
        avg_fit = sum(self.history[-10:]) / len(self.history[-10:])
        logger.info(
            f"Gen {self.current_generation + 1}: Best={self.best_solution[1]:.3f}, "
            f"Avg(last 10)={avg_fit:.3f}"
        )

    def reset(self) -> None:
        """Reset algorithm state for a new run."""
        self.population.clear()
        self.best_solution = None
        self.history.clear()
        self.current_generation = 0
        logger.debug("Algorithm state has been reset.")
