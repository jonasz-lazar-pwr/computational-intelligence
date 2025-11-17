from copy import deepcopy
from typing import List

from src.core.logger import get_logger
from src.core.models import ExperimentConfig
from src.factories.algorithm_factory import AlgorithmFactory
from src.factories.problem_factory import ProblemFactory
from src.interfaces.core_interfaces import IExperimentRunner, IResultCollector

logger = get_logger(__name__)


class ExperimentRunner(IExperimentRunner):
    """Runs experiment configurations and collects results."""

    def __init__(self, collector: IResultCollector) -> None:
        """Initialize the runner with a result collector."""
        self._collector = collector

    def run_all(self, configs: List[ExperimentConfig]) -> None:
        """Execute all provided experiment configurations."""
        if not configs:
            logger.warning("No experiment configurations to run.")
            return

        logger.info(f"Starting execution of {len(configs)} experiment configurations.")
        for cfg in configs:
            try:
                self._run_single(cfg)
            except Exception as e:
                logger.error(f"Error during execution of {cfg.name}: {e}", exc_info=True)

        logger.info("All experiments completed successfully.")

    def _run_single(self, cfg: ExperimentConfig) -> None:
        """Execute a single experiment configuration."""
        logger.info(f"Running experiment: {cfg.name}")

        problem_args = deepcopy(cfg.problem)
        problem_name = problem_args.pop("name", None)
        problem_args.pop("instance_name", None)

        if not isinstance(problem_name, str):
            raise ValueError(f"Invalid problem name: {cfg.problem}")

        problem = ProblemFactory.build(problem_name, **problem_args)

        for run_id in range(1, cfg.runs + 1):
            logger.debug(f"→ Run {run_id}/{cfg.runs} for {cfg.name}")

            seed = cfg.seed_base + run_id

            algo_args = dict(cfg.algorithm)
            algo_name = algo_args.pop("name")

            algo = AlgorithmFactory.build(
                algo_name,
                problem=problem,
                seed=seed,
                **algo_args,
            )

            result = algo.run()
            best_cost = result.get("best_cost")

            self._collector.collect_run(cfg.name, best_cost)

        optimal = getattr(problem, "optimal_value", None)
        optimal_value = optimal() if callable(optimal) else None

        self._collector.finalize_config(cfg.name, optimal_value, cfg.runs)
