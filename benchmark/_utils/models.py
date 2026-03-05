from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class RunConfig:
    """
    All resolved runtime settings for a benchmark run.
    Produced by the CLI/config parsing layer and consumed by the pipeline.
    """

    target: str
    dataset: str
    output: str
    params: List[Any] = field(default_factory=list)
    min_rounds: int = 5
    max_time: float = 0.05
    warmup: str = "off"
    warmup_iterations: int = 1
    force_rerun: bool = False
