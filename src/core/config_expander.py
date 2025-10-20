from itertools import product

from src.core.logger import get_logger
from src.core.models import ExperimentConfig
from src.interfaces.core_interfaces import IConfigExpander, IConfigValidator, INameGenerator

logger = get_logger(__name__)


class ConfigExpander(IConfigExpander):
    """Expands parsed YAML data into ExperimentConfig instances."""

    def __init__(self, validator: IConfigValidator, namer: INameGenerator):
        """Initialize with validator and name generator."""
        self._validator = validator
        self._namer = namer

    def expand(self, data: dict) -> list[ExperimentConfig]:
        """Expand YAML data into experiment configurations."""
        if "experiments" in data:
            return self._manual(data["experiments"])
        if "sweep" in data:
            return self._sweep(data["sweep"])
        logger.warning("No valid experiment section found in configuration.")
        return []

    def _manual(self, experiments: list[dict]) -> list[ExperimentConfig]:
        """Expand manually defined experiments."""
        configs = []
        for exp in experiments:
            self._validator.validate_problem(exp["problem"])
            self._validator.validate_algorithm(exp["algorithm"], allow_lists=False)
            cfg = ExperimentConfig(
                name=exp["name"],
                runs=exp["runs"],
                seed_base=exp["seed_base"],
                problem=exp["problem"],
                algorithm=exp["algorithm"],
            )
            configs.append(cfg)
        logger.info(f"Expanded {len(configs)} manual experiment configurations.")
        return configs

    def _sweep(self, sweeps: list[dict]) -> list[ExperimentConfig]:
        """Expand parameter sweeps into all combinations."""
        configs = []
        for sweep in sweeps:
            prefix = sweep["name_prefix"]
            runs = sweep["runs"]
            seed_base = sweep["seed_base"]
            problem = sweep["problem"]
            algorithm = sweep["algorithm"]
            self._validator.validate_problem(problem)

            base_keys = {k: v for k, v in algorithm.items() if not isinstance(v, list)}
            list_keys = {k: v for k, v in algorithm.items() if isinstance(v, list)}

            for values in product(*list_keys.values()):
                alg_cfg = base_keys.copy()
                alg_cfg.update(dict(zip(list_keys.keys(), values, strict=False)))
                self._validator.validate_algorithm(alg_cfg, allow_lists=False)
                name = self._namer.generate(problem, alg_cfg, prefix)
                cfg = ExperimentConfig(
                    name=name,
                    runs=runs,
                    seed_base=seed_base,
                    problem=problem,
                    algorithm=alg_cfg,
                )
                configs.append(cfg)
        logger.info(f"Expanded {len(configs)} sweep experiment configurations.")
        return configs
