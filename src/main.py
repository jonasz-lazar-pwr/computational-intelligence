import json

from pathlib import Path

from src.core.logger import get_logger
from src.factories.algorithm_factory import AlgorithmFactory
from src.factories.operator_factory import OperatorFactory
from src.factories.problem_factory import ProblemFactory

BASE_DIR = Path(__file__).resolve().parent.parent
logger = get_logger(__name__)


def main() -> None:
    """Run Genetic Algorithm with factory-based configuration."""
    logger.info(f"Running from base directory: {BASE_DIR}")

    tsp_args = {
        "file_path": str(BASE_DIR / "data" / "tsplib" / "ulysses16.tsp"),
        "optimal_results_path": str(BASE_DIR / "data" / "optimal_results.json"),
    }
    problem = ProblemFactory.build("tsp", **tsp_args)

    operator_factory = OperatorFactory()
    algorithm_config = {
        "problem": problem,
        "operator_factory": operator_factory,
        "population_size": 100,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "max_time": 10.0,
        "selection_config": {"name": "tournament", "tournament_size": 6},
        "crossover_config": {"name": "ox"},
        "mutation_config": {"name": "insert"},
    }

    algorithm = AlgorithmFactory.build("ga", **algorithm_config)
    result = algorithm.run()
    logger.info("Optimization results:\n%s", json.dumps(result, indent=2))
    logger.info("Optimization complete.")


if __name__ == "__main__":
    main()
