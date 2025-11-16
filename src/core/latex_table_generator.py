from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logger import get_logger
from src.interfaces.core_interfaces import ILatexTableGenerator

logger = get_logger(__name__)


class LatexTableGenerator(ILatexTableGenerator):
    """Generate LaTeX tables from experiment CSV data."""

    def __init__(self) -> None:
        """Initialize internal DataFrame cache."""
        self._df: pd.DataFrame | None = None

    @staticmethod
    def _load_csv(csv_path: Path) -> pd.DataFrame:
        """Load and sort the CSV file."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "mean_error" not in df.columns:
            raise ValueError("CSV file missing required column: mean_error")

        df = df.sort_values("mean_error", ascending=True)
        logger.info(f"Loaded and sorted {len(df)} records from {csv_path.name}")
        return df

    @staticmethod
    def _combine(name: str, param: float | None) -> str:
        """Combine operator name with parameter."""
        if pd.isna(param) or param == 0:
            return name
        return f"{name} ({param:.2f})"

    def _format_row(self, index: int, row: Any) -> dict[str, Any]:
        """Format a single CSV row for LaTeX."""
        return {
            "no": index + 1,
            "pop": int(row["population"]),
            "selection": self._combine(str(row["selection"]), row["sel_param"]),
            "cross": self._combine(str(row["crossover"]), row["cross_param"]),
            "mut": self._combine(str(row["mutation"]), row["mut_param"]),
            "succ": self._combine(str(row["succession"]), row["succ_param"]),
            "error": f"{row['mean_error'] * 100:.2f}\\%",
            "best": int(row["best_cost"]),
        }

    def _prepare_dataframe(self, df: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """Prepare formatted top-N results."""
        rows = [self._format_row(i, row) for i, (_, row) in enumerate(df.head(top_n).iterrows())]
        formatted_df = pd.DataFrame(rows)
        logger.info(f"Prepared {len(formatted_df)} rows for LaTeX export.")
        return formatted_df

    @staticmethod
    def _to_latex(df: pd.DataFrame) -> str:
        """Convert formatted DataFrame to LaTeX table."""
        latex_body = df.to_latex(
            index=False, column_format="c l l l l l c c", header=True, escape=False
        )

        full_table = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\caption{Najlepsze konfiguracje algorytmu genetycznego dla problemu TSP}\n"
            "\\label{tab:best_results}\n"
            f"{latex_body}"
            "\\end{table}\n"
        )
        return full_table

    def generate(self, csv_path: Path, output_path: Path, top_n: int) -> None:
        """Generate a LaTeX table and save it to file."""
        df = self._load_csv(csv_path)
        formatted_df = self._prepare_dataframe(df, top_n)
        latex_code = self._to_latex(formatted_df)
        output_path.write_text(latex_code, encoding="utf-8")
        logger.info(f"LaTeX table exported to {output_path}")
        self._df = formatted_df

    def get_dataframe(self) -> pd.DataFrame:
        """Return the last generated DataFrame."""
        if self._df is None:
            raise ValueError("No LaTeX data generated yet. Call generate() first.")
        return self._df
