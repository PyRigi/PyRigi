"""
Command-line entry point for the benchmarking framework.

The target and dataset paths are resolved against the current working directory,
so the command can be run from anywhere as long as they match. The bundled YAML
configs write theirs relative to the repository root, so run those from there:

    python benchmark/benchmarkapp.py --config benchmark/benchmark_config_random_2nm3.yaml

The same run given directly on the command line, from the repository root:

    python benchmark/benchmarkapp.py pyrigi/graph/_rigidity/generic.py:is_min_rigid \\
        --dataset benchmark/graph_store/random_2nm3 \\
        --params dim=2 algorithm=sparsity,randomized

or equivalently from inside benchmark/:

    python benchmarkapp.py ../pyrigi/graph/_rigidity/generic.py:is_min_rigid \\
        --dataset graph_store/random_2nm3 \\
        --params dim=2 algorithm=sparsity,randomized

Relative output paths are the exception: they are always anchored to benchmark/,
wherever the command is run from. See _utils/cli.py for the full option list.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = Path(__file__).resolve().parent

from _utils.cli import build_arg_parser, parse_and_resolve  # noqa: E402
from pipeline import run_benchmark_pipeline  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Parse the command line and run the benchmark pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = parse_and_resolve(args, BENCHMARK_DIR)

    run_benchmark_pipeline(config)


if __name__ == "__main__":
    main()
