from src.interfaces.core_interfaces import INameGenerator


class NameGenerator(INameGenerator):
    """Generate clean, consistent experiment names for GA and ACS."""

    @staticmethod
    def _sanitize(value):
        """Convert value to a safe string form."""
        return str(value).replace(".", "_")

    def _require(self, alg: dict, key: str):
        """Return required algorithm parameter or raise a clear error."""
        if key not in alg:
            raise ValueError(f"Missing required algorithm parameter: '{key}'")
        return alg[key]

    def generate(self, problem: dict, alg: dict) -> str:
        """Generate a standardized experiment name for a given problem and algorithm."""
        problem_name = problem.get("name", "unknown")
        instance = problem.get("instance_name", "unknown_instance")

        algo_name = alg.get("name", None)
        if algo_name not in ("ga", "acs"):
            raise ValueError(f"Unknown algorithm type in NameGenerator: {algo_name}")

        if algo_name == "ga":
            pop = self._require(alg, "population_size")
            max_time = int(float(self._require(alg, "max_time")))

            sel = self._require(alg, "selection_config")
            cx = self._require(alg, "crossover_config")
            mut = self._require(alg, "mutation_config")
            succ = self._require(alg, "succession_config")

            sel_name = sel.get("name", "sel")
            sel_rate = self._sanitize(sel.get("rate", 0))

            cx_name = cx.get("name", "cx")
            cx_rate = self._sanitize(alg.get("crossover_rate", 0))

            mut_name = mut.get("name", "mut")
            mut_rate = self._sanitize(alg.get("mutation_rate", 0))

            succ_name = succ.get("name", "succ")
            succ_rate = self._sanitize(succ.get("elite_rate") or succ.get("replacement_rate") or 0)

            return (
                f"{problem_name}_{instance}_ga_"
                f"population_{pop}_"
                f"time_{max_time}_"
                f"{sel_name}_{sel_rate}_"
                f"{cx_name}_{cx_rate}_"
                f"{mut_name}_{mut_rate}_"
                f"{succ_name}_{succ_rate}"
            )

        if algo_name == "acs":
            max_time = int(float(self._require(alg, "max_time")))

            return (
                f"{problem_name}_{instance}_acs_"
                f"ants_{self._sanitize(self._require(alg, 'num_ants'))}_"
                f"alpha_{self._sanitize(self._require(alg, 'alpha'))}_"
                f"beta_{self._sanitize(self._require(alg, 'beta'))}_"
                f"rho_{self._sanitize(self._require(alg, 'rho'))}_"
                f"phi_{self._sanitize(self._require(alg, 'phi'))}_"
                f"q0_{self._sanitize(self._require(alg, 'q0'))}_"
                f"time_{max_time}"
            )

        raise ValueError(f"Unhandled algorithm type: {algo_name}")
