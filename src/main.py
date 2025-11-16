from pathlib import Path

from src.core.config_expander import ConfigExpander
from src.core.config_loader import ConfigLoader
from src.core.config_service import ConfigService
from src.core.config_validator import ConfigValidator
from src.core.experiment_runner import ExperimentRunner
from src.core.logger import disable_file_logging, get_logger
from src.core.name_generator import NameGenerator
from src.core.result_collector import ResultCollector
from src.core.statistics import Statistics

BASE_DIR = Path(__file__).resolve().parent.parent
logger = get_logger(__name__)


def main() -> None:
    """Load configurations and execute all experiments."""
    disable_file_logging()

    logger.info(f"Running experiments from base directory: {BASE_DIR}")

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


if __name__ == "__main__":
    main()

    # results_path = output_dir / "results.json"
    # parser = ResultParser(results_path)
    # parser.load()
    # parser.parse()
    # parser.export_csv(output_dir / "results_parsed.csv")
    #
    # parser_output = output_dir / "results_parsed.csv"
    # latex_output = output_dir / "results_table.tex"
    #
    # table_generator = LatexTableGenerator()
    # table_generator.generate(parser_output, latex_output, 40)

    # plots = ComparisonPlotGenerator()
    #
    # plots.generate_selection_by_population(
    #     parser_output, BASE_DIR / "results" / "fig" / "sel_vs_pop.png"
    # )
    # plots.generate_crossover_by_succession(
    #     parser_output, BASE_DIR / "results" / "fig" / "cross_vs_succ.png"
    # )
    # plots.generate_mutation_by_selection(
    #     parser_output, BASE_DIR / "results" / "fig" / "mut_vs_sel.png"
    # )
    # plots.generate_succession_vs_selection_heatmap(
    #     parser_output, BASE_DIR / "results" / "fig" / "succ_vs_sel.png"
    # )
