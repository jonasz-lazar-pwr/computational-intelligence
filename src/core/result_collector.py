import json

from pathlib import Path
from typing import List, Tuple

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IResultCollector, IStatistics

logger = get_logger(__name__)


class ResultCollector(IResultCollector):
    """Collects and aggregates experiment results."""

    def __init__(self, output_dir: str | Path, statistics: IStatistics):
        """Initialize with output directory and statistics module."""
        self._base_path = Path(output_dir)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._statistics = statistics
        self._results_cache: dict[str, list[float]] = {}

    def collect_run(self, config_name: str, history: List[Tuple[float, float]]) -> None:
        """Store single algorithm run results."""
        if not history:
            logger.warning(f"No history recorded for {config_name}. Skipping run save.")
            return
        run_dir = self._base_path / config_name
        run_dir.mkdir(exist_ok=True)
        run_index = len(list(run_dir.glob("run_*.json"))) + 1
        run_path = run_dir / f"run_{run_index}.json"
        data = {
            "config_name": config_name,
            "history": [(float(t), float(c)) for t, c in history],
        }
        with run_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)  # type: ignore[arg-type]
        best_cost = min(cost for _, cost in history)
        self._results_cache.setdefault(config_name, []).append(best_cost)
        logger.debug(f"Saved run #{run_index} for {config_name} (best={best_cost:.3f}).")

    def finalize_config(self, config_name: str, optimal_value: float | None, runs: int) -> None:
        """Aggregate results and save summary file."""
        run_dir = self._base_path / config_name
        run_dir.mkdir(exist_ok=True)
        results = self._results_cache.get(config_name, [])
        if not results:
            logger.warning(f"No runs recorded for {config_name}. Summary skipped.")
            return
        mean_error = self._statistics.compute_mean_error(results, optimal_value)
        best_cost = self._statistics.best_cost(results)
        summary = {
            "config_name": config_name,
            "runs": runs,
            "mean_error": float(mean_error),
            "best_cost": float(best_cost),
        }
        summary_path = run_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)  # type: ignore[arg-type]
        self._results_cache.pop(config_name, None)
        logger.info(f"Finalized {config_name}: mean_error={mean_error:.5f}, best={best_cost:.3f}")
