import os
import sys
import json
import datetime
from pathlib import Path

from _utils import (
    function_loader,
    dataset_loader,
    param_parser,
    benchmark_merger,
    test_generator_mult,
    runner,
)
from _utils.models import RunConfig

BENCHMARK_DIR = Path(__file__).resolve().parent


def _validate_function_parameters(func, graph_param_name, configs):
    """
    Validate that the generated configurations match the target
    function's signature.
    """
    if not (configs and configs[0]):
        return
    import inspect

    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys()) - {graph_param_name}

    all_config_params = set()
    for config in configs:
        all_config_params.update(config.keys())

    invalid_params = all_config_params - valid_params
    if invalid_params:
        print(
            f"Error: Invalid parameter name(s) detected: "
            f"{', '.join(sorted(invalid_params))}"
        )
        print(
            f"Valid parameters for '{func.__name__}': "
            f"{', '.join(sorted(valid_params))}"
        )
        print("Hint: Check your config file or --params argument for typos.")
        sys.exit(1)

    print(f"Parameter validation passed: {', '.join(sorted(all_config_params))}")


def _filter_existing_benchmarks(
    output: str, force_rerun: bool, func_name: str, dataset_paths: list, configs: list
):
    """Returns (existing_results, configs_to_run, explicit_tasks)"""
    existing_results = benchmark_merger.load_existing_results(output)

    if force_rerun:
        print(
            f"Force-rerun enabled. Existing results for '{func_name}' "
            "will be overwritten."
        )
        backup_path = benchmark_merger.create_backup(output)
        print(f"Backup created at: {backup_path}")

        return existing_results, configs, None

    print("Checking for existing results to avoid redundancy...")

    graph_infos = dataset_loader.load_graph_infos(dataset_paths, func_name)

    missing_combinations, num_skipped = benchmark_merger.filter_existing_combinations(
        configs, graph_infos, existing_results, func_name
    )

    if num_skipped > 0:
        print(f"Skipping {num_skipped} existing configuration/graph pairs.")

    if not missing_combinations:
        print("All requested benchmarks already exist. Nothing to run.")
        sys.exit(0)

    print(f"Running {len(missing_combinations)} specific benchmark tasks.")
    return existing_results, configs, missing_combinations


def run_benchmark_pipeline(config: RunConfig) -> None:
    """Execute the full benchmarking pipeline based on the provided configuration."""
    timeout_str = f"{config.timeout}s" if config.timeout else "none"
    print(
        f"Benchmark settings: min_rounds={config.min_rounds}, "
        f"max_time={config.max_time}s, "
        f"warmup={config.warmup}, warmup_iterations={config.warmup_iterations}, "
        f"timeout={timeout_str}"
    )

    # 1. Load Function & Detect Graph Parameter
    print(f"Loading function from {config.target}...")
    try:
        func, graph_param_name = function_loader.load_function_and_detect_param(
            config.target
        )
        print(
            f"Found function '{func.__name__}' with graph parameter '{graph_param_name}'"
        )
    except Exception as e:
        print(f"Error loading function: {e}")
        sys.exit(1)

    # 2. Load Dataset
    print(f"Loading dataset from {config.dataset}...")
    try:
        dataset_paths = dataset_loader.get_dataset_files(config.dataset)
        print(f"Found {len(dataset_paths)} graph files.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if not dataset_paths:
        print("No .g6 files found in dataset directory.")
        sys.exit(1)

    # 3. Parse Parameters
    print("Parsing parameters...")

    if (
        isinstance(config.params, list)
        and len(config.params) > 0
        and isinstance(config.params[0], dict)
    ):
        # Already parsed by YAML loader
        configs = config.params
        print(
            f"Loaded {len(configs)} configuration(s) from config file "
            f"(cartesian + explicit)."
        )
    else:
        # Needs parsing from CLI strings
        raw_params = param_parser.parse_cli_strings(config.params)
        configs = param_parser.build_cartesian_product(raw_params)
        print(
            f"Generated {len(configs)} configuration(s) from CLI "
            f"using Cartesian product strategy."
        )

    # 3.5. Validate Parameters Against Function Signature
    _validate_function_parameters(func, graph_param_name, configs)

    # 4. Filter Existing Benchmarks
    if os.path.exists(config.output):
        print(f"Loading existing results from {config.output}...")
        existing_results, configs_to_run, explicit_tasks = _filter_existing_benchmarks(
            config.output, config.force_rerun, func.__name__, dataset_paths, configs
        )
    else:
        existing_results = {}
        configs_to_run = configs
        explicit_tasks = None
        print(f"Creating new results file: {config.output}")

    # Paths for temp files
    test_file = str(BENCHMARK_DIR / "temp_benchmark_test.py")
    temp_results_file = str(BENCHMARK_DIR / "temp_benchmark_results.json")
    conftest_file = str(BENCHMARK_DIR / "conftest.py")
    timeout_log_file = str(BENCHMARK_DIR / "timeout_log.json")

    # 5. Generate Test File First (to compute nodes_total)
    print("Generating temporary test file...")
    try:
        test_file, nodes_total = test_generator_mult.generate_benchmark_test_file(
            func_path=config.target,
            func_name=func.__name__,
            graph_param_name=graph_param_name,
            dataset_paths=dataset_paths,
            configurations=configs_to_run,
            explicit_combinations=explicit_tasks,
            output_path=test_file,
        )
        print(f"Test file generated at {test_file}")
    except Exception as e:
        print(f"Error generating test file: {e}")
        sys.exit(1)

    # 6. Generate conftest.py
    print("Generating conftest.py (data-strip hook + timeout support)...")
    try:
        test_generator_mult.generate_conftest_file(
            str(BENCHMARK_DIR),
            timeout_seconds=config.timeout,
            timeout_threshold=config.timeout_threshold,
            nodes_total=nodes_total,
        )
        print(f"conftest.py generated at {conftest_file}")
    except Exception as e:
        print(f"Error generating conftest.py: {e}")
        sys.exit(1)

    # 7. Run Benchmarks
    print("Running benchmarks...")
    try:
        runner.run_pytest_benchmark(
            test_file,
            temp_results_file,
            min_rounds=config.min_rounds,
            max_time=config.max_time,
            warmup=config.warmup,
            warmup_iterations=config.warmup_iterations,
            timeout=config.timeout,
            timeout_log_file=timeout_log_file,
        )
        print(f"Benchmarks completed. Results in {temp_results_file}")

        # 8. Load temp results & enrich
        if os.path.exists(temp_results_file):
            with open(temp_results_file, "r") as f:
                new_results = json.load(f)

            timestamp = datetime.datetime.now().isoformat()
            source_hash = function_loader.get_function_hash(func)
            print(f"Computed source hash for {func.__name__}: {source_hash}")

            for benchmark in new_results.get("benchmarks", []):
                benchmark["function"] = func.__name__
                benchmark["module_path"] = config.target
                benchmark["timestamp"] = timestamp
                benchmark["source_hash"] = source_hash

            # 9. Merge and Write
            final_results = benchmark_merger.merge_results(
                existing_results, new_results, func.__name__, config.force_rerun
            )

            with open(config.output, "w") as f:
                json.dump(final_results, f, indent=2)

            print(f"Results successfully merged and saved to {config.output}")
        else:
            print("Warning: No results generated (maybe no tests were run?)")

        # 10. Process timeout log
        _process_timeout_log(timeout_log_file, config)

    except Exception as e:
        print(f"Benchmark run failed: {e}")
    finally:
        for tmp in (test_file, conftest_file, temp_results_file, timeout_log_file):
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass


def _process_timeout_log(timeout_log_file: str, config: RunConfig) -> None:
    """Read the timeout log, print a summary, and save timeout_results.json."""
    if not os.path.exists(timeout_log_file):
        return

    with open(timeout_log_file, "r") as f:
        timeout_data = json.load(f)

    timed_out = timeout_data.get("timed_out", [])
    timeout_seconds = timeout_data.get("timeout_seconds")
    early_stops = timeout_data.get("early_stops", [])

    if not timed_out and not early_stops:
        print("No test cases timed out or stopped early.")
        return

    print(f"\n{'=' * 60}")
    print(
        f"TIMEOUT SUMMARY: {len(timed_out)} test(s) timed out (limit: {timeout_seconds}s)"
    )
    if early_stops:
        print(f"EARLY STOPS: {len(early_stops)} config(s) hit timeout threshold")
    print(f"{'=' * 60}")
    for nodeid in timed_out:
        print(f"  - {nodeid}")
    for stop in early_stops:
        print(
            f"  - Config {stop['config']} stopped at n={stop['stopped_at_num_nodes']} "
            f"({stop['timeout_count']}/{stop['total_count']} timeouts)"
        )
    print(f"{'=' * 60}\n")

    # Save to a separate timeout_results.json
    output_dir = Path(config.output).parent
    timeout_results_path = str(output_dir / "timeout_results.json")

    # Load existing timeout results
    existing_timeouts = []
    existing_early_stops = []
    if os.path.exists(timeout_results_path):
        with open(timeout_results_path, "r") as f:
            existing_data = json.load(f)
        existing_timeouts = existing_data.get("timeouts", [])
        existing_early_stops = existing_data.get("early_stops", [])

    timestamp = datetime.datetime.now().isoformat()
    new_entries = [
        {
            "nodeid": nodeid,
            "function": config.target.split(":")[-1],
            "timeout_seconds": timeout_seconds,
            "timestamp": timestamp,
        }
        for nodeid in timed_out
    ]

    for stop in early_stops:
        stop["function"] = config.target.split(":")[-1]
        stop["timestamp"] = timestamp

    all_timeouts = existing_timeouts + new_entries
    all_early_stops = existing_early_stops + early_stops

    with open(timeout_results_path, "w") as f:
        json.dump(
            {
                "timeouts": all_timeouts,
                "early_stops": all_early_stops,
            },
            f,
            indent=2,
        )

    print(f"Timeout results saved to {timeout_results_path}")
