from pathlib import Path

from src.core.comparison_plot_generator_acs import ACSComparisonPlotGenerator
from src.core.config_expander import ConfigExpander
from src.core.config_loader import ConfigLoader
from src.core.config_service import ConfigService
from src.core.config_validator import ConfigValidator
from src.core.experiment_runner import ExperimentRunner
from src.core.latex_table_generator_acs import LatexTableGeneratorACS
from src.core.logger import get_logger
from src.core.name_generator import NameGenerator
from src.core.result_collector import ResultCollector
from src.core.result_parser_acs import ResultParserACS
from src.core.statistics import Statistics

BASE_DIR = Path(__file__).resolve().parent.parent
logger = get_logger(__name__)


def main() -> None:
    """Load configurations and execute all experiments."""
    logger.info("All experiments completed successfully.")

    # === Configuration setup ===
    config_path = BASE_DIR / "config" / "experiments_config.yaml"
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = ConfigValidator()
    loader = ConfigLoader(str(config_path), validator)
    expander = ConfigExpander(validator, NameGenerator())
    config_service = ConfigService(loader, validator, expander)

    configs = config_service.load_all()
    if not configs:
        logger.warning("No valid experiment configurations found. Exiting.")
        return

    # === Statistics & result management ===
    statistics = Statistics()
    collector = ResultCollector(output_dir, statistics)

    # === Run experiments ===
    runner = ExperimentRunner(collector)
    runner.run_all(configs)

    logger.info("All experiments completed successfully.")

    results_path = Path("results/results.json")

    parser = ResultParserACS(results_path)
    parser.load()
    parser.parse()
    parser.export_csv(Path("results/acs_results.csv"))

    gen = LatexTableGeneratorACS()
    gen.generate(Path("results/acs_results.csv"), Path("results/acs_table.tex"), top_n=40)

    plots = ACSComparisonPlotGenerator()

    plots.plot_alpha_vs_beta(
        Path("results/acs_results.csv"),
        Path("results/fig/acs_alpha_vs_beta.png"),
    )

    plots.plot_rho_vs_phi(
        Path("results/acs_results.csv"),
        Path("results/fig/acs_rho_vs_phi.png"),
    )

    plots.plot_q0_vs_beta(
        Path("results/acs_results.csv"),
        Path("results/fig/acs_q0_vs_beta.png"),
    )

    plots.plot_ants_vs_rho(
        Path("results/acs_results.csv"),
        Path("results/fig/acs_ants_vs_rho.png"),
    )

    plots.plot_param_heatmap(
        Path("results/acs_results.csv"), Path("results/fig/acs_param_global_influence.png")
    )


if __name__ == "__main__":
    main()
