from pathlib import Path
from typing import List, Optional

from src.core.logger import get_logger
from src.interfaces.i_tsp_catalog import ITSPCatalog
from src.interfaces.i_tsp_instance import ITSPInstance
from src.problems.tsp.tsp_instance import TSPInstance

logger = get_logger(__name__)


class TSPCatalog(ITSPCatalog):
    """Manage a collection of TSP instances."""

    def __init__(self, optimal_results_path: str) -> None:
        """Initialize catalog with a path to optimal results."""
        self.instances: List[ITSPInstance] = []
        self.optimal_results_path: Path = Path(optimal_results_path)

    def clear_files(self) -> None:
        """Clear all loaded TSP instances."""
        count = len(self.instances)
        self.instances.clear()
        logger.debug(f"Cleared {count} TSP instances from memory.")

    def load_files(self, directory_path: str) -> None:
        """Load and parse all .tsp files from a directory."""
        self.instances.clear()
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        logger.info(f"Loading TSP instances from directory: {directory.resolve()}")

        for file_path in sorted(directory.iterdir()):
            if file_path.suffix != ".tsp":
                logger.debug(f"Skipping non-TSP file: {file_path.name}")
                continue
            try:
                instance: ITSPInstance = TSPInstance(
                    file_path=str(file_path),
                    optimal_results_path=str(self.optimal_results_path),
                )
                instance.load_metadata()
                self.instances.append(instance)
                logger.info(f"Loaded TSP instance: {instance.name} ({instance.dimension} cities)")
            except FileNotFoundError as e:
                logger.warning(f"File not found: {file_path.name} — {e}")
            except ValueError as e:
                logger.warning(f"Invalid TSP format in {file_path.name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error loading {file_path.name}: {e}", exc_info=True)

        logger.info(f"Finished loading {len(self.instances)} TSP instances.")

    def get_file_by_name(self, name: str) -> Optional[ITSPInstance]:
        """Return a TSP instance by name or file name."""
        try:
            for instance in self.instances:
                if instance.name == name or str(instance.file_path).endswith(f"{name}.tsp"):
                    logger.debug(f"Found TSP instance by name: {name}")
                    return instance
        except Exception as e:
            logger.error(f"Error while retrieving instance '{name}': {e}", exc_info=True)
        return None

    def list_instances(self) -> List[str]:
        """Return list of loaded TSP instance names."""
        names = [inst.name for inst in self.instances if inst.name]
        logger.debug(f"Listing {len(names)} loaded TSP instance names.")
        return names

    def summary(self) -> None:
        """Log summary of all loaded instances."""
        if not self.instances:
            logger.info("No TSP instances loaded.")
            return
        logger.info(f"Loaded {len(self.instances)} TSP instances:")
        for inst in self.instances:
            logger.info(f" - {inst.name} ({inst.dimension} cities, type={inst.edge_weight_type})")
