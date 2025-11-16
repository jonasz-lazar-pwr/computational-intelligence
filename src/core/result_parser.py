import json

from pathlib import Path
from typing import Any, List

import pandas as pd

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IResultParser

logger = get_logger(__name__)


class ResultParser(IResultParser):
    """Parse and process experiment results JSON into a structured DataFrame."""

    def __init__(self, results_path: Path):
        """Initialize parser with path to results.json."""
        if not results_path.exists():
            raise FileNotFoundError(f"results file not found: {results_path}")
        self._results_path = results_path
        self._data: List[dict[str, Any]] = []
        self._df: pd.DataFrame | None = None

    def load(self) -> None:
        """Load raw JSON data into memory."""
        with self._results_path.open("r", encoding="utf-8") as f:
            self._data = json.load(f)
        logger.info(f"Loaded {len(self._data)} result entries from {self._results_path.name}")

    def _parse_config_name(self, name: str) -> dict[str, Any]:
        """Extract parameters and operators from a config name."""
        parts = name.split("_")

        try:
            population = int(parts[parts.index("population") + 1])
        except (ValueError, IndexError):
            population = None

        try:
            start = parts.index("time") + 2
        except ValueError:
            start = 0
        tokens = parts[start:]

        merged: list[str] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == "steady" and tokens[i + 1] == "state":
                merged.append("steady_state")
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        ops: list[tuple[str, float | None]] = []
        i = 0
        while i < len(merged):
            op = merged[i]
            val = None
            if i + 2 < len(merged) and merged[i + 1].isdigit() and merged[i + 2].isdigit():
                val = float(f"{merged[i + 1]}.{merged[i + 2]}")
                i += 3
            elif i + 1 < len(merged) and merged[i + 1].replace(".", "", 1).isdigit():
                val = float(merged[i + 1])
                i += 2
            else:
                i += 1
            ops.append((op, val))

        idx_selection = 0
        idx_crossover = 1
        idx_mutation = 2
        idx_succession = 3

        return {
            "population": population,
            "selection": ops[idx_selection][0] if len(ops) > idx_selection else None,
            "sel_param": ops[idx_selection][1] if len(ops) > idx_selection else None,
            "crossover": ops[idx_crossover][0] if len(ops) > idx_crossover else None,
            "cross_param": ops[idx_crossover][1] if len(ops) > idx_crossover else None,
            "mutation": ops[idx_mutation][0] if len(ops) > idx_mutation else None,
            "mut_param": ops[idx_mutation][1] if len(ops) > idx_mutation else None,
            "succession": ops[idx_succession][0] if len(ops) > idx_succession else None,
            "succ_param": ops[idx_succession][1] if len(ops) > idx_succession else None,
        }

    def parse(self) -> None:
        """Convert loaded results JSON into a DataFrame."""
        if not self._data:
            raise ValueError("No data loaded. Call load() first.")
        rows = []
        for entry in self._data:
            params = self._parse_config_name(entry["config_name"])
            if params:
                params["mean_error"] = entry["mean_error"]
                params["best_cost"] = entry["best_cost"]
                rows.append(params)
        df = pd.DataFrame(rows)
        self._df = self._standardize_labels(df)
        logger.info(f"Parsed results into DataFrame with {len(self._df)} rows.")

    @staticmethod
    def _standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize operator labels for readability."""
        replacements = {
            "selection": {"tournament": "tournament", "roulette": "roulette"},
            "crossover": {"ox": "ox", "pmx": "pmx"},
            "mutation": {"insert": "insert", "swap": "swap"},
            "succession": {
                "elitist": "elitist",
                "steady_state": "steady",
                "steady": "steady",
                "state": "steady",
            },
        }
        for col, mapping in replacements.items():
            df[col] = df[col].replace(mapping)
        return df

    def export_csv(self, output_path: Path) -> None:
        """Export parsed DataFrame as a sorted CSV."""
        if self._df is None:
            raise ValueError("No parsed data to export. Run parse() first.")

        df = self._df.copy()
        df["mean_error"] = df["mean_error"].apply(lambda x: round(float(x), 4))
        df = df.sort_values(by="mean_error", ascending=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Exported sorted and formatted results to {output_path}")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the parsed DataFrame."""
        if self._df is None:
            raise ValueError("No parsed data available. Run parse() first.")
        return self._df
