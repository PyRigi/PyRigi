"""
The settings object shared by every stage of the benchmark pipeline.

RunConfig is built once by the CLI/config layer (see cli.parse_and_resolve) and
passed unchanged to pipeline.run_benchmark_pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional


@dataclass
class RunConfig:
    """
    All resolved runtime settings for a benchmark run.
    Produced by the CLI/config parsing layer and consumed by the pipeline.

    Attributes:
        target: Function under test, as "path/to/file.py:function_name".
        dataset: Directory containing the .g6 graph files to benchmark against.
        output: Path to the JSON results file that results are merged into.
        params: Either a list of explicit config dicts (from a YAML config) or a
            list of "key=val1,val2" CLI strings expanded into a Cartesian product.
        min_rounds: Minimum pytest-benchmark rounds per test case.
        max_time: Time budget in seconds for pytest-benchmark's adaptive round
            loop per test case.
        warmup: pytest-benchmark warmup mode, one of "auto", "on" or "off".
        warmup_iterations: Warmup rounds used when warmup is "on" or "auto".
        force_rerun: Discard existing results for this function and measure again.
        timeout: Per-test-case timeout in seconds, or None to disable.
        timeout_threshold: Fraction of timeouts at one vertex count that stops a
            configuration from being attempted at larger sizes, or None to
            disable early stopping.
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
    timeout: Optional[float] = None
    timeout_threshold: Optional[float] = None
