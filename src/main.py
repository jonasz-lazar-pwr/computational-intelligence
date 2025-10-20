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
