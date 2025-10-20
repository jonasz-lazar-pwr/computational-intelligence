from pathlib import Path

from src.interfaces.core_interfaces import INameGenerator


class NameGenerator(INameGenerator):
    """Builds consistent experiment identifiers from configuration dictionaries."""

    @staticmethod
    def _sanitize(value: float | int | str) -> str:
        """Convert value to string and replace '.' with '_'."""
        return str(value).replace(".", "_")

    def generate(self, problem: dict, alg: dict, prefix: str) -> str:
        """Generate a unique experiment name based on problem and algorithm config."""
        instance = Path(problem["file_path"]).stem
        sel = alg["selection_config"]["name"]
        sel_rate = alg["selection_config"].get("rate", 0)
        cx = alg["crossover_config"]["name"]
        cx_rate = alg.get("crossover_rate", 0)
        mut = alg["mutation_config"]["name"]
        mut_rate = alg.get("mutation_rate", 0)
        succ = alg["succession_config"]["name"]
        succ_rate = (
            alg["succession_config"].get("elite_rate")
            or alg["succession_config"].get("replacement_rate")
            or 0
        )
        return (
            f"{prefix}_tsp_{instance}_ga_population_{alg['population_size']}_"
            f"time_{int(alg['max_time'])}_{sel}_{self._sanitize(sel_rate)}_"
            f"{cx}_{self._sanitize(cx_rate)}_{mut}_{self._sanitize(mut_rate)}_"
            f"{succ}_{self._sanitize(succ_rate)}"
        )
