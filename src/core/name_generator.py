from src.interfaces.core_interfaces import INameGenerator


class NameGenerator(INameGenerator):
    """Generates clean, consistent experiment names based on problem and algorithm config."""

    @staticmethod
    def _sanitize(value: float | int | str) -> str:
        """Convert value to string and replace '.' with '_'."""
        return str(value).replace(".", "_")

    def generate(self, problem: dict, alg: dict) -> str:
        """Generate experiment name using fixed scheme."""
        problem_name = problem.get("name", "unknown")
        instance = problem.get("instance_name") or "unknown_instance"

        pop_size = alg.get("population_size", 0)
        max_time = alg.get("max_time", 0)
        algo_name = alg.get("name", "unknown")

        sel_cfg = alg.get("selection_config", {})
        sel_name = sel_cfg.get("name", "sel")
        sel_rate = sel_cfg.get("rate", 0)

        cx_cfg = alg.get("crossover_config", {})
        cx_name = cx_cfg.get("name", "cx")
        cx_rate = alg.get("crossover_rate", 0)

        mut_cfg = alg.get("mutation_config", {})
        mut_name = mut_cfg.get("name", "mut")
        mut_rate = alg.get("mutation_rate", 0)

        succ_cfg = alg.get("succession_config", {})
        succ_name = succ_cfg.get("name", "succ")
        succ_rate = succ_cfg.get("elite_rate") or succ_cfg.get("replacement_rate") or 0

        return (
            f"{problem_name}_{instance}_"
            f"{algo_name}_population_{pop_size}_"
            f"time_{int(float(max_time))}_"
            f"{sel_name}_{self._sanitize(sel_rate)}_"
            f"{cx_name}_{self._sanitize(cx_rate)}_"
            f"{mut_name}_{self._sanitize(mut_rate)}_"
            f"{succ_name}_{self._sanitize(succ_rate)}"
        )
