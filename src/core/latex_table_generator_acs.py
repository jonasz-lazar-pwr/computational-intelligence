from pathlib import Path

import pandas as pd

from src.core.logger import get_logger

logger = get_logger(__name__)


class LatexTableGeneratorACS:
    """Generates LaTeX tables for ACS best configurations."""

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load and sort ACS results CSV."""
        df = pd.read_csv(path)
        return df.sort_values("mean_error")

    @staticmethod
    def _fmt(x: float) -> str:
        """Format parameter value to two decimals."""
        return f"{float(x):.2f}"

    @staticmethod
    def _format_row(i: int, row: pd.Series) -> dict:
        """Format a single ACS result row for LaTeX."""
        return {
            "no": i + 1,
            "ants": int(row["num_ants"]),
            "alpha": LatexTableGeneratorACS._fmt(row["alpha"]),
            "beta": LatexTableGeneratorACS._fmt(row["beta"]),
            "rho": LatexTableGeneratorACS._fmt(row["rho"]),
            "phi": LatexTableGeneratorACS._fmt(row["phi"]),
            "q0": LatexTableGeneratorACS._fmt(row["q0"]),
            "error": f"{row['mean_error'] * 100:.2f}\\%",
            "best": int(row["best_cost"]),
        }

    def generate(self, csv_path: Path, output_path: Path, top_n: int) -> None:
        """Generate LaTeX table with top-N ACS configurations."""
        df = self._load_csv(csv_path)
        rows = [self._format_row(i, row) for i, row in df.head(top_n).iterrows()]
        table_df = pd.DataFrame(rows)

        latex = table_df.to_latex(
            index=False,
            escape=False,
            column_format="c c c c c c c c c",
        )

        out = (
            "\\begin{table}[H]\n"
            "\\centering\n"
            "\\caption{Najlepsze konfiguracje algorytmu ACS}\n"
            "\\label{tab:acs_best_results}\n"
            f"{latex}"
            "\\end{table}\n"
        )

        output_path.write_text(out, encoding="utf-8")
        logger.info(f"ACS LaTeX table saved to {output_path}")
