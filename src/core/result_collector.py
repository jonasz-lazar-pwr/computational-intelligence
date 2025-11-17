import json
import statistics as py_stats

from pathlib import Path

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IResultCollector, IStatistics

logger = get_logger(__name__)


class ResultCollector(IResultCollector):
    """Collects and aggregates experiment run results."""

    def __init__(self, output_dir: str | Path, statistics: IStatistics):
        """Initialize collector with output directory and statistics handler."""
        self._base_path = Path(output_dir)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._results_path = self._base_path / "results.json"
        self._statistics = statistics
        self._results_cache: dict[str, list[float]] = {}

    def collect_run(self, config_name: str, best_cost: float) -> None:
        """Store best_cost from a single run."""
        if best_cost is None:
            logger.warning(f"No best_cost for {config_name} — skipping run.")
            return
        self._results_cache.setdefault(config_name, []).append(best_cost)
        logger.debug(f"Collected run for {config_name}: best={best_cost}")

    def finalize_config(self, config_name: str, optimal_value: float | None, runs: int) -> None:
        """Compute statistics and write configuration results to results.json."""
        results = self._results_cache.get(config_name, [])
        if not results:
            logger.warning(f"No collected results for {config_name}. Skipping.")
            return

        min_cost = min(results)
        max_cost = max(results)
        mean_cost = sum(results) / len(results)
        std_cost = py_stats.stdev(results) if len(results) > 1 else 0.0

        if optimal_value:
            errors = [(c - optimal_value) / optimal_value for c in results]
            mean_error = sum(errors) / len(errors)
            success_rate = sum(1 for e in errors if e <= 0.01) / len(errors)
        else:
            mean_error = 0.0
            success_rate = 0.0

        entry = {
            "config_name": config_name,
            "runs": runs,
            "min_cost": float(min_cost),
            "max_cost": float(max_cost),
            "mean_cost": float(mean_cost),
            "std_cost": float(std_cost),
            "mean_error": float(mean_error),
            "success_rate": float(success_rate),
        }

        try:
            if self._results_path.exists():
                with self._results_path.open("r", encoding="utf-8") as f:
                    results_data = json.load(f)
                if not isinstance(results_data, list):
                    results_data = []
            else:
                results_data = []
        except Exception:
            results_data = []

        results_data.append(entry)

        with self._results_path.open("w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, sort_keys=True, ensure_ascii=False)

        self._results_cache.pop(config_name, None)
