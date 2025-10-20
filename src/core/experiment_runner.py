from copy import deepcopy
from typing import List

from src.core.logger import get_logger
from src.core.models import ExperimentConfig
from src.factories.algorithm_factory import AlgorithmFactory
from src.factories.problem_factory import ProblemFactory
from src.interfaces.core_interfaces import IExperimentRunner, IResultCollector

logger = get_logger(__name__)


class ExperimentRunner(IExperimentRunner):
    """Executes all experiment configurations and aggregates results into a global summary."""

    def __init__(self, collector: IResultCollector) -> None:
        """Initialize with a shared result collector."""
        self._collector = collector

    def run_all(self, configs: List[ExperimentConfig]) -> None:
        """Run all experiments and aggregate their results."""
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
        """Run a single experiment configuration for the specified number of runs."""
        logger.info(f"Running experiment: {cfg.name}")

        problem_args = deepcopy(cfg.problem)
        problem_name = problem_args.pop("name", None)
        problem_args.pop("instance_name", None)
        if not isinstance(problem_name, str):
            raise ValueError(f"Invalid or missing problem name in config: {cfg.problem}")

        problem = ProblemFactory.build(problem_name, **problem_args)

        for run_id in range(1, cfg.runs + 1):
            logger.debug(f"→ Run {run_id}/{cfg.runs} for {cfg.name}")
            seed = cfg.seed_base + run_id

            algo_args = dict(cfg.algorithm)
            algo_name = algo_args.pop("name", None)
            if not isinstance(algo_name, str):
                raise ValueError(f"Invalid or missing algorithm name in config: {cfg.algorithm}")

            algo = AlgorithmFactory.build(
                algo_name,
                problem=problem,
                seed=seed,
                **algo_args,
            )
            result = algo.run()
            history = result.get("history", [])
            self._collector.collect_run(cfg.name, history)

        optimal_func = getattr(problem, "optimal_value", None)
        optimal_value = optimal_func() if callable(optimal_func) else None

        self._collector.finalize_config(cfg.name, optimal_value, cfg.runs)
        logger.info(f"Finished {cfg.name} and appended to summary.")
