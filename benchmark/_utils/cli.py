import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from _utils import config_loader
from _utils.models import RunConfig

BENCHMARK_DIR = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for benchmarkapp."""
    parser = argparse.ArgumentParser(description="PyRigi Centralized Benchmarking Tool")

    parser.add_argument("--config", help="Path to YAML config file (optional)")

    parser.add_argument(
        "target",
        nargs="?",
        help="Target function in format: path/to/file.py:function_name",
    )

    parser.add_argument(
        "--dataset", help="Path to directory containing .g6 graph files"
    )

    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help="Parameters in format name=val1,val2 (e.g., dim=1,2 algo=A,B)",
    )

    parser.add_argument("--output", default=None, help="Path to output JSON file")

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run benchmarks for current function, ignoring existing results",
    )

    parser.add_argument(
        "--min-rounds",
        type=int,
        default=None,
        help=(
            "Minimum number of benchmark rounds per test case "
            "(default: 5 from config, or 5 if not set). "
            "Lower values are appropriate for complexity analysis."
        ),
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help=(
            "Maximum time (in seconds) to spend on each benchmark test case's "
            "adaptive round loop (default: 0.05). "
            "pytest-benchmark runs at least min_rounds, then keeps adding rounds "
            "until max_time is exhausted."
        ),
    )

    parser.add_argument(
        "--warmup",
        default=None,
        choices=["auto", "on", "off"],
        help=(
            "Warmup mode for pytest-benchmark "
            "(default: 'off' from config, or 'off' if not set). "
            "Use 'auto' to let pytest-benchmark decide."
        ),
    )

    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=None,
        help=(
            "Number of warmup rounds when --warmup is 'on' or 'auto' "
            "(default: 1). Has no effect when --warmup=off."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help=(
            "Per-test-case timeout in seconds. If a single benchmark "
            "test case runs longer than this, it is skipped and recorded "
            "in a separate timeout results file. "
            "Default: no timeout."
        ),
    )

    parser.add_argument(
        "--timeout-threshold",
        type=float,
        default=None,
        help=(
            "Early-stop threshold (0.0-1.0). If the fraction of timeouts for a "
            "specific config and vertex size reaches this, testing stops for "
            "larger graphs of that config. Default: no early stop."
        ),
    )

    return parser


def _load_config_file(args) -> Dict[str, Any]:
    """Load and merge YAML config file if provided."""
    if not args.config:
        return {}
    print(f"Loading configuration from {args.config}...")
    try:
        cfg = config_loader.load_config(args.config)
        config_loader.validate_config(cfg)
        return config_loader.merge_with_cli(cfg, args)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def _validate_cli_only(args) -> None:
    """Validate required CLI args when no config file is provided."""
    if not args.target:
        sys.stderr.write("error: target is required when not using --config\n")
        sys.exit(1)
    if not args.dataset:
        sys.stderr.write("error: --dataset is required when not using --config\n")
        sys.exit(1)


def parse_and_resolve(args, benchmark_dir: Path) -> RunConfig:
    """
    Resolve CLI args + optional config file into a RunConfig.

    Args:
        args: Parsed argparse namespace.
        benchmark_dir: Absolute path to the benchmark/ directory (used to
                       anchor relative output paths).

    Returns:
        Fully resolved RunConfig instance.
    """
    merged_config = _load_config_file(args)

    if merged_config:
        target = merged_config.get("target")
        dataset = merged_config.get("dataset")
        output = merged_config.get(
            "output", str(benchmark_dir / "benchmark_results.json")
        )

        # If the YAML config loaded combinations via cartesian/explicit keys,
        # they are merged under "configurations". If not, fall back to cli string params.
        if "configurations" in merged_config:
            params = merged_config["configurations"]
        else:
            params = merged_config.get("cli_params", [])
    else:
        _validate_cli_only(args)
        target = args.target
        dataset = args.dataset
        output = (
            args.output
            if args.output
            else str(benchmark_dir / "benchmark_results.json")
        )
        params = args.params if args.params else []

    # Anchor relative output paths to benchmark_dir
    output_path = Path(output)
    if not output_path.is_absolute():
        output = str(benchmark_dir / output_path)

    min_rounds = args.min_rounds
    if min_rounds is None:
        min_rounds = merged_config.get("min_rounds", 5)

    max_time = args.max_time
    if max_time is None:
        max_time = merged_config.get("max_time", 0.05)

    warmup = args.warmup
    if warmup is None:
        warmup = merged_config.get("warmup", "off")

    warmup_iterations = args.warmup_iterations
    if warmup_iterations is None:
        warmup_iterations = merged_config.get("warmup_iterations", 1)

    timeout = args.timeout
    if timeout is None:
        timeout = merged_config.get("timeout", None)

    timeout_threshold = args.timeout_threshold
    if timeout_threshold is None:
        timeout_threshold = merged_config.get("timeout_threshold", None)

    return RunConfig(
        target=target,
        dataset=dataset,
        output=output,
        params=params,
        min_rounds=min_rounds,
        max_time=max_time,
        warmup=warmup,
        warmup_iterations=warmup_iterations,
        force_rerun=args.force_rerun,
        timeout=timeout,
        timeout_threshold=timeout_threshold,
    )
