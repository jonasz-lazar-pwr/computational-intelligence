import json

from pathlib import Path
from typing import Any, List

import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class ResultParserACS:
    """Parse ACS experiment results into a structured DataFrame."""

    def __init__(self, results_path: Path):
        """Initialize parser with path to results.json."""
        if not results_path.exists():
            raise FileNotFoundError(f"results file not found: {results_path}")
        self._results_path = results_path
        self._data: List[dict[str, Any]] = []
        self._df: pd.DataFrame | None = None

    def load(self) -> None:
        """Load raw JSON data."""
        with self._results_path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)
        logger.info(f"Loaded {len(self._data)} ACS entries.")

    @staticmethod
    def _parse_number(parts: list[str], key: str) -> float:
        """Convert tokens like X_Y into float X.Y."""
        if key not in parts:
            return None
        i = parts.index(key)
        whole = parts[i + 1]
        frac = parts[i + 2]
        return float(f"{whole}.{frac}")

    @staticmethod
    def _parse_config_name(name: str) -> dict[str, Any]:
        """Extract ACS parameter values from config_name."""
        parts = name.split("_")

        num_ants = int(parts[parts.index("ants") + 1])
        alpha = float(parts[parts.index("alpha") + 1].replace("_", "."))
        beta = float(parts[parts.index("beta") + 1].replace("_", "."))

        rho = ResultParserACS._parse_number(parts, "rho")
        phi = ResultParserACS._parse_number(parts, "phi")
        q0 = ResultParserACS._parse_number(parts, "q0")

        return {
            "num_ants": num_ants,
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "phi": phi,
            "q0": q0,
        }

    def parse(self) -> None:
        """Transform raw entries into a sorted DataFrame."""
        rows = []
        for entry in self._data:
            params = self._parse_config_name(entry["config_name"])
            params["mean_error"] = entry["mean_error"]
            params["best_cost"] = entry["best_cost"]
            rows.append(params)

        df = pd.DataFrame(rows)
        df = df.sort_values("mean_error")
        self._df = df
        logger.info(f"Parsed ACS data into DataFrame with {len(df)} rows.")

    def export_csv(self, output_path: Path) -> None:
        """Save parsed DataFrame as CSV."""
        if self._df is None:
            raise ValueError("Parse data before exporting.")
        self._df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"ACS CSV exported to {output_path}")

    def get_dataframe(self) -> pd.DataFrame:
        """Return parsed DataFrame."""
        if self._df is None:
            raise ValueError("No parsed data available.")
        return self._df
