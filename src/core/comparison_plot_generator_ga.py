from pathlib import Path
from typing import ClassVar, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.core.logger import get_logger
from src.interfaces.core_interfaces import IComparisonPlotGenerator

logger = get_logger(__name__)


class GAComparisonPlotGenerator(IComparisonPlotGenerator):
    """Generate comparison plots for algorithm performance metrics."""

    _PALETTE: ClassVar[list[str]] = ["#1f77b4", "#ff7f0e"]

    def __init__(self) -> None:
        """Initialize internal dataframe storage."""
        self._df: pd.DataFrame | None = None

    def _load(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV file and validate required columns."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        req = {
            "population",
            "selection",
            "sel_param",
            "crossover",
            "cross_param",
            "succession",
            "mean_error",
            "mutation",
            "mut_param",
        }
        missing = req.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        logger.info(f"Loaded {len(df)} rows from {csv_path.name}")
        return df

    @staticmethod
    def _label_with_param(name: str, param: float | int | None, fmt: str = ".2f") -> str:
        """Return formatted label including parameter if provided."""
        if pd.isna(param) or float(param) == 0:
            return str(name)
        return f"{name} ({param:{fmt}})"

    @staticmethod
    def _set_common_style(
        xlabel: str, ylabel: str, legend_title: str, legend_loc: str = "upper left"
    ) -> None:
        """Configure common plot labels and layout."""
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(title=legend_title, loc=legend_loc)
        plt.tight_layout()

    @staticmethod
    def _order_categories(series: Iterable[str], desired: list[str]) -> list[str]:
        """Order categories by a preferred order while preserving unknown items."""
        seen = [x for x in desired if x in set(series)]
        rest = [x for x in series if x not in desired]
        return list(dict.fromkeys(seen + rest))

    def _aggregate_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean error by selection method and population size."""
        g = (
            df.groupby(["selection", "sel_param", "population"], dropna=False)
            .mean(numeric_only=True)[["mean_error"]]
            .reset_index()
        )
        g["mean_error"] *= 100.0
        g["selection_label"] = g.apply(
            lambda r: self._label_with_param(r["selection"], r["sel_param"]), axis=1
        )
        logger.info(f"Selection grouped to {len(g)} rows.")
        return g

    def _plot_selection(self, g: pd.DataFrame, output_path: Path) -> None:
        """Plot selection vs population comparison chart."""
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=g,
            x="selection_label",
            y="mean_error",
            hue="population",
            palette=self._PALETTE,
        )
        plt.title("")
        self._set_common_style(
            xlabel="Metoda selekcji",
            ylabel="Średni błąd [%]",
            legend_title="Rozmiar populacji",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Selection plot saved to {output_path}")

    def generate_selection_by_population(self, csv_path: Path, output_path: Path) -> None:
        """Generate selection operator comparison plot by population."""
        df = self._load(csv_path)
        g = self._aggregate_selection(df)
        desired = ["roulette", "tournament (0.03)", "tournament (0.07)"]
        order = self._order_categories(g["selection_label"], desired)
        g["selection_label"] = pd.Categorical(g["selection_label"], categories=order, ordered=True)
        self._plot_selection(g.sort_values(["selection_label", "population"]), output_path)

    def _aggregate_crossover_by_succession(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean error by crossover operator and succession strategy."""
        g = (
            df.groupby(["crossover", "cross_param", "succession"], dropna=False)
            .mean(numeric_only=True)[["mean_error"]]
            .reset_index()
        )
        g["mean_error"] *= 100.0
        g["cross_label"] = g.apply(
            lambda r: self._label_with_param(str(r["crossover"]).upper(), r["cross_param"]),
            axis=1,
        )
        logger.info(f"Crossover x Succession grouped to {len(g)} rows.")
        return g

    def _plot_crossover_by_succession(self, g: pd.DataFrame, output_path: Path) -> None:
        """Plot crossover vs succession comparison chart."""
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=g,
            x="cross_label",
            y="mean_error",
            hue="succession",
            palette=self._PALETTE,
        )
        plt.title("")
        self._set_common_style(
            xlabel="Operator krzyżowania",
            ylabel="Średni błąd [%]",
            legend_title="Strategia sukcesji",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Crossover x Succession plot saved to {output_path}")

    def generate_crossover_by_succession(self, csv_path: Path, output_path: Path) -> None:
        """Generate crossover operator comparison plot by succession strategy."""
        df = self._load(csv_path)
        g = self._aggregate_crossover_by_succession(df)
        desired = ["OX (0.85)", "OX (0.95)", "PMX (0.85)", "PMX (0.95)"]
        order = self._order_categories(g["cross_label"], desired)
        g["cross_label"] = pd.Categorical(g["cross_label"], categories=order, ordered=True)
        self._plot_crossover_by_succession(
            g.sort_values(["cross_label", "succession"]), output_path
        )

    def _aggregate_mutation_by_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean error by mutation operator and selection method."""
        g = (
            df.groupby(["mutation", "mut_param", "selection"], dropna=False)
            .mean(numeric_only=True)[["mean_error"]]
            .reset_index()
        )
        g["mean_error"] *= 100.0
        g["mut_label"] = g.apply(
            lambda r: self._label_with_param(r["mutation"], r["mut_param"]), axis=1
        )
        logger.info(f"Mutation x Selection grouped to {len(g)} rows.")
        return g

    def _plot_mutation_by_selection(self, g: pd.DataFrame, output_path: Path) -> None:
        """Plot mutation vs selection comparison chart."""
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=g,
            x="mut_label",
            y="mean_error",
            hue="selection",
            palette=self._PALETTE,
        )
        plt.title("")
        self._set_common_style(
            xlabel="Operator mutacji",
            ylabel="Średni błąd [%]",
            legend_title="Metoda selekcji",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Mutation x Selection plot saved to {output_path}")

    def generate_mutation_by_selection(self, csv_path: Path, output_path: Path) -> None:
        """Generate mutation operator comparison plot by selection method."""
        df = self._load(csv_path)
        g = self._aggregate_mutation_by_selection(df)
        desired = ["insert (0.02)", "insert (0.05)", "swap (0.02)", "swap (0.05)"]
        order = self._order_categories(g["mut_label"], desired)
        g["mut_label"] = pd.Categorical(g["mut_label"], categories=order, ordered=True)
        self._plot_mutation_by_selection(g.sort_values(["mut_label", "selection"]), output_path)

    def _aggregate_succession_vs_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean error by succession strategy and selection method."""
        g = (
            df.groupby(["succession", "succ_param", "selection"], dropna=False)
            .mean(numeric_only=True)[["mean_error"]]
            .reset_index()
        )
        g["mean_error"] *= 100.0
        g["succ_label"] = g.apply(
            lambda r: self._label_with_param(r["succession"], r["succ_param"]),
            axis=1,
        )
        g["sel_label"] = g["selection"].astype(str)
        logger.info(f"Succession x Selection grouped to {len(g)} rows.")
        return g

    def _plot_succession_vs_selection_heatmap(self, g: pd.DataFrame, output_path: Path) -> None:
        """Plot heatmap comparing succession and selection methods."""
        pivot = g.pivot(index="sel_label", columns="succ_label", values="mean_error")

        plt.figure(figsize=(8, 4))
        ax = sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar=True,
            annot_kws={"fontsize": 9},
        )

        for t in ax.texts:
            val = t.get_text()
            t.set_text(f"{val}%")

        colorbar = ax.collections[0].colorbar
        colorbar.set_label("Średni błąd [%]")
        colorbar.set_ticks(colorbar.get_ticks())
        colorbar.set_ticklabels([f"{v:.0f}%" for v in colorbar.get_ticks()])

        plt.xlabel("Strategia sukcesji")
        plt.ylabel("Metoda selekcji")
        plt.xticks(rotation=0)
        plt.yticks(rotation=90)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Succession x Selection heatmap saved to {output_path}")

    def generate_succession_vs_selection_heatmap(self, csv_path: Path, output_path: Path) -> None:
        """Generate heatmap for succession vs selection comparison."""
        df = self._load(csv_path)
        g = self._aggregate_succession_vs_selection(df)

        desired_succ = ["elitist (0.03)", "elitist (0.07)", "steady (0.03)", "steady (0.07)"]
        desired_sel = ["roulette", "tournament"]

        g["succ_label"] = pd.Categorical(
            g["succ_label"],
            categories=self._order_categories(g["succ_label"], desired_succ),
            ordered=True,
        )
        g["sel_label"] = pd.Categorical(
            g["sel_label"],
            categories=self._order_categories(g["sel_label"], desired_sel),
            ordered=True,
        )

        self._plot_succession_vs_selection_heatmap(g, output_path)
