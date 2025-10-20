from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    runs: int
    seed_base: int
    problem: dict[str, Any]
    algorithm: dict[str, Any]
