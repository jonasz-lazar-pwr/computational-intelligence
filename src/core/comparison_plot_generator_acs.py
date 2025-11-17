from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.core.logger import get_logger

logger = get_logger(__name__)


class ACSComparisonPlotGenerator:
    """Generator for ACS parameter comparison plots."""

    PALETTE: ClassVar[list[str]] = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    @staticmethod
    def _load(csv_path: Path) -> pd.DataFrame:
        """Load ACS CSV results and validate required columns."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        req = {"num_ants", "alpha", "beta", "rho", "phi", "q0", "mean_error"}
        missing = req.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        df["mean_error"] *= 100.0
        return df

    @staticmethod
    def _fmt(symbol: str, value: float) -> str:
        """Format parameter label for plotting."""
        return f"{symbol} = {value:.2f}" if isinstance(value, float) else f"{symbol} = {value}"

    @staticmethod
    def _save(fig_path: Path) -> None:
        """Save the current matplotlib figure to file."""
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Plot saved to {fig_path}")

    def plot_alpha_vs_beta(self, csv_path: Path, output_path: Path) -> None:
        """Plot mean error for combinations of alpha and beta."""
        df = self._load(csv_path)
        g = df.groupby(["alpha", "beta"]).mean(numeric_only=True)[["mean_error"]].reset_index()
        g["alpha_label"] = g["alpha"].apply(lambda x: self._fmt("α", x))
        g["beta_label"] = g["beta"].apply(lambda x: self._fmt("β", x))

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=g,
            x="alpha_label",
            y="mean_error",
            hue="beta_label",
            palette=self.PALETTE,
        )
        plt.xlabel("Parametr α (waga feromonu)")
        plt.ylabel("Średni błąd względny [%]")
        plt.legend(title="Parametr β (waga heurystyki)")

        self._save(output_path)

    def plot_rho_vs_phi(self, csv_path: Path, output_path: Path) -> None:
        """Plot mean error for combinations of rho and phi."""
        df = self._load(csv_path)
        g = df.groupby(["rho", "phi"]).mean(numeric_only=True)[["mean_error"]].reset_index()

        g["rho_label"] = g["rho"].apply(lambda x: self._fmt("ρ", x))
        g["phi_label"] = g["phi"].apply(lambda x: self._fmt("ϕ", x))

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=g,
            x="rho_label",
            y="mean_error",
            hue="phi_label",
            palette=self.PALETTE,
        )
        plt.xlabel("Parametr globalnego parowania ρ")
        plt.ylabel("Średni błąd względny [%]")
        plt.legend(title="Parametr lokalnego parowania ϕ")
        self._save(output_path)

    def plot_q0_vs_beta(self, csv_path: Path, output_path: Path) -> None:
        """Plot mean error for combinations of q0 and beta."""
        df = self._load(csv_path)
        g = df.groupby(["q0", "beta"]).mean(numeric_only=True)[["mean_error"]].reset_index()

        g["q0_label"] = g["q0"].apply(lambda x: self._fmt("q₀", x))
        g["beta_label"] = g["beta"].apply(lambda x: self._fmt("β", x))

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=g,
            x="q0_label",
            y="mean_error",
            hue="beta_label",
            palette=self.PALETTE,
        )
        plt.xlabel("Parametr eksploracji–eksploatacji q₀")
        plt.ylabel("Średni błąd względny [%]")
        plt.legend(title="Parametr β")
        self._save(output_path)

    def plot_ants_vs_rho(self, csv_path: Path, output_path: Path) -> None:
        """Plot mean error for combinations of number of ants and rho."""
        df = self._load(csv_path)
        g = df.groupby(["num_ants", "rho"]).mean(numeric_only=True)[["mean_error"]].reset_index()

        g["ants_label"] = g["num_ants"].apply(lambda x: self._fmt("m", x))
        g["rho_label"] = g["rho"].apply(lambda x: self._fmt("ρ", x))

        plt.figure(figsize=(9, 5))
        sns.barplot(
            data=g,
            x="ants_label",
            y="mean_error",
            hue="rho_label",
            palette=self.PALETTE,
        )
        plt.xlabel("Liczba mrówek m")
        plt.ylabel("Średni błąd względny [%]")
        plt.legend(title="Parametr ρ")
        self._save(output_path)

    def plot_param_heatmap(self, csv_path: Path, output_path: Path) -> None:
        """Plot heatmap showing mean error for each ACS parameter value."""
        df = self._load(csv_path)

        symbols = {
            "alpha": "α",
            "beta": "β",
            "rho": "ρ",
            "phi": "ϕ",
            "q0": "q₀",
        }

        rows = []
        for param, symbol in symbols.items():
            grp = df.groupby(param).mean(numeric_only=True)["mean_error"]
            for val, err in grp.items():
                rows.append({"param": symbol, "value": val, "mean_error": err})

        long_df = pd.DataFrame(rows)
        pivot = long_df.pivot_table(index="value", columns="param", values="mean_error")
        annot = pivot.copy().map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            pivot,
            annot=annot,
            fmt="",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": "Średni błąd [%]"},
        )

        cbar = ax.collections[0].colorbar
        vmin, vmax = pivot.min().min(), pivot.max().max()
        ticks = np.linspace(vmin, vmax, 5)

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.0f}%" for t in ticks])

        plt.xlabel("Parametr")
        plt.ylabel("Wartości parametrów")
        plt.title("")

        self._save(output_path)
