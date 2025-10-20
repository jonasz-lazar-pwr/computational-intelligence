from typing import List

from src.core.logger import get_logger
from src.core.models import ExperimentConfig
from src.factories.algorithm_factory import AlgorithmFactory
from src.factories.problem_factory import ProblemFactory
from src.interfaces.core_interfaces import IExperimentRunner, IResultCollector

logger = get_logger(__name__)


class ExperimentRunner(IExperimentRunner):
    """Executes multiple experiment configurations using factories and collectors."""

    def __init__(self, collector: IResultCollector) -> None:
        """Initialize with result collector."""
        self._collector = collector

    def run_all(self, configs: List[ExperimentConfig]) -> None:
        """Run all experiments and delegate results to collector."""
        if not configs:
            logger.warning("No experiment configurations to run.")
            return
        logger.info(f"Starting execution of {len(configs)} experiment configurations.")
        for cfg in configs:
            try:
                self._run_single(cfg)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"Error during execution of {cfg.name}: {e}", exc_info=True)
        logger.info("All experiments completed successfully.")

    def _run_single(self, cfg: ExperimentConfig) -> None:
        """Run a single experiment configuration across multiple runs."""
        logger.info(f"Running experiment: {cfg.name}")
        problem = ProblemFactory.build(cfg.problem["name"], **cfg.problem)
        for run_id in range(1, cfg.runs + 1):
            logger.debug(f"→ Run {run_id}/{cfg.runs} for {cfg.name}")
            seed = cfg.seed_base + run_id
            algo = AlgorithmFactory.build(
                cfg.algorithm["name"],
                problem=problem,
                seed=seed,
                **cfg.algorithm,
            )
            result = algo.run()
            history = result.get("history", [])
            self._collector.collect_run(cfg.name, history)
        optimal_func = getattr(problem, "optimal_value", None)
        optimal = optimal_func() if callable(optimal_func) else None
        self._collector.finalize_config(cfg.name, optimal, cfg.runs)
