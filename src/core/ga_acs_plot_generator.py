import json

from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class GAACSComparisonPlotGenerator:
    """Generates comparison plots for GA and ACS reference performance."""

    PALETTE: ClassVar[list[str]] = ["#1f77b4", "#ff7f0e"]

    def __init__(self) -> None:
        """Initialize internal GA/ACS storage."""
        self.ga: dict | None = None
        self.acs: dict | None = None

    @staticmethod
    def _load_results(results_path: Path) -> tuple[dict, dict]:
        """Load GA and ACS results from results.json."""
        if not results_path.exists():
            raise FileNotFoundError(f"results.json not found: {results_path}")

        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        ga = None
        acs = None

        for entry in data:
            name = entry["config_name"]
            if name.startswith("tsp_bays29_ga"):
                ga = entry
            elif name.startswith("tsp_bays29_acs"):
                acs = entry

        if ga is None or acs is None:
            raise RuntimeError("Missing GA or ACS reference entries in results.json")

        logger.info("Loaded reference GA and ACS metrics from results.json")
        return ga, acs

    @staticmethod
    def _save(fig_path: Path) -> None:
        """Save the current matplotlib figure."""
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Plot saved to {fig_path}")

    def plot_mean_cost(self, results_path: Path, output_path: Path) -> None:
        """Plot mean cost with standard deviation for GA and ACS."""
        ga, acs = self._load_results(results_path)

        df = pd.DataFrame(
            {
                "Algorithm": ["GA", "ACS"],
                "mean_cost": [ga["mean_cost"], acs["mean_cost"]],
                "std_cost": [ga["std_cost"], acs["std_cost"]],
            }
        )

        plt.figure(figsize=(7, 5))
        plt.bar(
            df["Algorithm"],
            df["mean_cost"],
            yerr=df["std_cost"],
            capsize=8,
            color=self.PALETTE,
        )

        plt.ylabel("Średni koszt")
        self._save(output_path)

    def plot_success_rate(self, results_path: Path, output_path: Path) -> None:
        """Plot success rate (≤1%) for GA and ACS."""
        ga, acs = self._load_results(results_path)

        df = pd.DataFrame(
            {
                "Algorithm": ["GA", "ACS"],
                "success_rate": [ga["success_rate"], acs["success_rate"]],
            }
        )

        plt.figure(figsize=(7, 5))
        plt.bar(df["Algorithm"], df["success_rate"], color=self.PALETTE)

        plt.ylabel("Odsetek wyników ≤ 1%")
        self._save(output_path)
