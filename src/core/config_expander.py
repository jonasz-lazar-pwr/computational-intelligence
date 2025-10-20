from copy import deepcopy
from itertools import product

from src.core.logger import get_logger
from src.core.models import ExperimentConfig
from src.interfaces.core_interfaces import IConfigExpander, IConfigValidator, INameGenerator

logger = get_logger(__name__)


class ConfigExpander(IConfigExpander):
    """Expands parsed YAML data into ExperimentConfig instances."""

    def __init__(self, validator: IConfigValidator, namer: INameGenerator):
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
            configs.append(
                ExperimentConfig(
                    name=exp["name"],
                    runs=exp["runs"],
                    seed_base=exp["seed_base"],
                    problem=exp["problem"],
                    algorithm=exp["algorithm"],
                )
            )
        logger.info(f"Expanded {len(configs)} manual experiment configurations.")
        return configs

    def _expand_nested_dict(self, d: dict) -> list[dict]:
        """Expand dict where some fields are lists."""
        if not d:
            return [{}]
        keys = list(d.keys())
        values = [v if isinstance(v, list) else [v] for v in d.values()]
        return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]

    def _expand_operator_section(self, section):
        """Expand operator section — supports dict or list of dicts."""
        if isinstance(section, dict):
            return self._expand_nested_dict(section)
        if isinstance(section, list):
            expanded = []
            for block in section:
                expanded.extend(self._expand_nested_dict(block))
            return expanded
        return [section]

    def _sweep(self, sweeps: list[dict]) -> list[ExperimentConfig]:
        """Expand parameter sweeps into all valid experiment configurations."""
        configs = []
        seen = set()

        for sweep in sweeps:
            runs = sweep["runs"]
            seed_base = sweep["seed_base"]
            problem = sweep["problem"]
            algorithm = deepcopy(sweep["algorithm"])

            self._validator.validate_problem(problem)

            # Expand operator sections
            for op_key in [
                "selection_config",
                "crossover_config",
                "mutation_config",
                "succession_config",
            ]:
                if op_key in algorithm:
                    algorithm[op_key] = self._expand_operator_section(algorithm[op_key])

            # Split scalar vs list params
            base_keys = {k: v for k, v in algorithm.items() if not isinstance(v, list)}
            list_keys = {k: v for k, v in algorithm.items() if isinstance(v, list)}

            for values in product(*list_keys.values()):
                alg_cfg = deepcopy(base_keys)
                alg_cfg.update(dict(zip(list_keys.keys(), values, strict=False)))

                # Combine operator configurations (Cartesian product)
                operator_products = product(
                    *[
                        algorithm[k]
                        for k in [
                            "selection_config",
                            "crossover_config",
                            "mutation_config",
                            "succession_config",
                        ]
                        if k in algorithm
                    ]
                )

                for op_combo in operator_products:
                    final_cfg = deepcopy(alg_cfg)
                    for i, k in enumerate(
                        [
                            "selection_config",
                            "crossover_config",
                            "mutation_config",
                            "succession_config",
                        ]
                    ):
                        if k in algorithm:
                            final_cfg[k] = op_combo[i]

                    # Deduplicate identical configs
                    key = str(sorted(final_cfg.items()))
                    if key in seen:
                        continue
                    seen.add(key)

                    self._validator.validate_algorithm(final_cfg, allow_lists=False)
                    name = self._namer.generate(problem, final_cfg)

                    configs.append(
                        ExperimentConfig(
                            name=name,
                            runs=runs,
                            seed_base=seed_base,
                            problem=problem,
                            algorithm=final_cfg,
                        )
                    )

        logger.info(f"Expanded {len(configs)} sweep experiment configurations.")
        return configs
