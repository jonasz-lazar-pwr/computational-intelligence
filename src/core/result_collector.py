import json

from pathlib import Path
from typing import List, Tuple

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IResultCollector, IStatistics

logger = get_logger(__name__)


class ResultCollector(IResultCollector):
    """Collects and aggregates experiment results into one summary.json file."""

    def __init__(self, output_dir: str | Path, statistics: IStatistics):
        self._base_path = Path(output_dir)
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._summary_path = self._base_path / "summary.json"
        self._statistics = statistics
        self._results_cache: dict[str, list[float]] = {}

    def collect_run(self, config_name: str, history: List[Tuple[float, float]]) -> None:
        """Cache best result from a single run (no file writing)."""
        if not history:
            logger.warning(f"No history recorded for {config_name}. Skipping run collection.")
            return

        best_cost = min(cost for _, cost in history)
        self._results_cache.setdefault(config_name, []).append(best_cost)
        logger.debug(f"Collected run for {config_name} (best={best_cost:.3f}).")

    def finalize_config(self, config_name: str, optimal_value: float | None, runs: int) -> None:
        """Compute statistics and append them to global summary.json."""
        results = self._results_cache.get(config_name, [])
        if not results:
            logger.warning(f"No runs recorded for {config_name}. Skipping summary append.")
            return

        mean_error = self._statistics.compute_mean_error(results, optimal_value)
        best_cost = self._statistics.best_cost(results)
        entry = {
            "config_name": config_name,
            "runs": runs,
            "mean_error": float(mean_error),
            "best_cost": float(best_cost),
        }

        try:
            if self._summary_path.exists():
                with self._summary_path.open("r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                if not isinstance(summary_data, list):
                    summary_data = []
            else:
                summary_data = []
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupted or unreadable summary.json — recreating file.")
            summary_data = []

        summary_data.append(entry)
        with self._summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, sort_keys=True, ensure_ascii=False)  # type: ignore[arg-type]

        logger.info(
            f"Finalized {config_name}: mean_error={mean_error:.5f}, best={best_cost:.3f} "
            f"(written to summary.json)"
        )

        self._results_cache.pop(config_name, None)
