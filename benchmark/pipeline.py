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
    checkpoint,
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


def _exit_with_error(prefix: str, error: Exception) -> None:
    print(f"{prefix}: {error}")
    sys.exit(1)


def _load_function_and_hash(config: RunConfig):
    print(f"Loading function from {config.target}...")
    try:
        func, graph_param_name = function_loader.load_function_and_detect_param(
            config.target
        )
        print(
            f"Found function '{func.__name__}' with graph parameter '{graph_param_name}'"
        )
    except Exception as e:
        _exit_with_error("Error loading function", e)

    source_hash = function_loader.get_function_hash(func)
    print(f"Source hash for {func.__name__}: {source_hash}")

    return func, graph_param_name, source_hash


def _load_dataset_paths(config: RunConfig) -> list:
    print(f"Loading dataset from {config.dataset}...")
    try:
        dataset_paths = dataset_loader.get_dataset_files(config.dataset)
        print(f"Found {len(dataset_paths)} graph files.")
    except Exception as e:
        _exit_with_error("Error loading dataset", e)

    if not dataset_paths:
        print("No .g6 files found in dataset directory.")
        sys.exit(1)

    return dataset_paths


def _parse_configurations(config: RunConfig) -> list:
    print("Parsing parameters...")

    if (
        isinstance(config.params, list)
        and len(config.params) > 0
        and isinstance(config.params[0], dict)
    ):
        parsed_configs = config.params
        print(
            f"Loaded {len(parsed_configs)} configuration(s) from config file "
            f"(cartesian + explicit)."
        )
        return parsed_configs

    raw_params = param_parser.parse_cli_strings(config.params)
    parsed_configs = param_parser.build_cartesian_product(raw_params)
    print(
        f"Generated {len(parsed_configs)} configuration(s) from CLI "
        f"using Cartesian product strategy."
    )
    return parsed_configs


def _recover_checkpoint_if_present(checkpoint_file: str, output: str) -> None:
    if not checkpoint.exists(checkpoint_file):
        return

    print("\nFound checkpoint from a previous interrupted run. Recovering...")
    try:
        n_recovered = checkpoint.drain_into_results(checkpoint_file, output)
        print(f"Recovered {n_recovered} result(s) into {output}.\n")
    except Exception as e:
        print(f"Warning: Checkpoint recovery failed ({e}). Continuing without it.")
        checkpoint.clear(checkpoint_file)


def _prepare_run_scope(
    config: RunConfig, func_name: str, dataset_paths: list, configs: list
):
    if os.path.exists(config.output):
        print(f"Loading existing results from {config.output}...")
        return _filter_existing_benchmarks(
            config.output, config.force_rerun, func_name, dataset_paths, configs
        )

    print(f"Creating new results file: {config.output}")
    return {}, configs, None


def _generate_test_file(
    config: RunConfig,
    func_name: str,
    graph_param_name: str,
    dataset_paths: list,
    configs_to_run: list,
    explicit_tasks,
    test_file: str,
    source_hash: str,
):
    print("Generating temporary test file...")
    try:
        return test_generator_mult.generate_benchmark_test_file(
            func_path=config.target,
            func_name=func_name,
            graph_param_name=graph_param_name,
            dataset_paths=dataset_paths,
            configurations=configs_to_run,
            explicit_combinations=explicit_tasks,
            output_path=test_file,
            source_hash=source_hash,
            min_rounds=config.min_rounds,
            max_time=config.max_time,
            warmup=config.warmup,
        )
    except Exception as e:
        _exit_with_error("Error generating test file", e)


def _generate_conftest(config: RunConfig, conftest_file: str, nodes_total: int) -> None:
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
        _exit_with_error("Error generating conftest.py", e)


def _cleanup_temp_files(*temp_paths: str) -> None:
    for tmp in temp_paths:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _enrich_results(
    temp_results_file: str, func_name: str, target: str, source_hash: str
) -> dict:
    with open(temp_results_file, "r") as f:
        new_results = json.load(f)

    timestamp = datetime.datetime.now().isoformat()
    for benchmark in new_results.get("benchmarks", []):
        benchmark["function"] = func_name
        benchmark["module_path"] = target
        benchmark["timestamp"] = timestamp
        benchmark["source_hash"] = source_hash

    return new_results


def _merge_and_write_results(
    config: RunConfig, existing_results: dict, new_results: dict, func_name: str
) -> None:
    final_results = benchmark_merger.merge_results(
        existing_results, new_results, func_name, config.force_rerun
    )

    tmp_output = config.output + ".tmp"
    with open(tmp_output, "w") as f:
        json.dump(final_results, f, indent=2)
    os.replace(tmp_output, config.output)

    print(f"Results successfully merged and saved to {config.output}")


def _run_and_persist_results(
    config: RunConfig,
    test_file: str,
    conftest_file: str,
    temp_results_file: str,
    timeout_log_file: str,
    checkpoint_file: str,
    existing_results: dict,
    func_name: str,
    source_hash: str,
) -> None:
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
            checkpoint_file=checkpoint_file,
        )
        print(f"Benchmarks completed. Results in {temp_results_file}")

        if os.path.exists(temp_results_file):
            new_results = _enrich_results(
                temp_results_file, func_name, config.target, source_hash
            )
            _merge_and_write_results(config, existing_results, new_results, func_name)
            checkpoint.clear(checkpoint_file)
        else:
            print("Warning: No results generated (maybe no tests were run?)")

        _process_timeout_log(timeout_log_file, config)
    except Exception as e:
        print(f"Benchmark run failed: {e}")
    finally:
        _cleanup_temp_files(
            test_file, conftest_file, temp_results_file, timeout_log_file
        )


def run_benchmark_pipeline(config: RunConfig) -> None:
    """Execute the full benchmarking pipeline based on the provided configuration."""
    timeout_str = f"{config.timeout}s" if config.timeout else "none"
    print(
        f"Benchmark settings: min_rounds={config.min_rounds}, "
        f"max_time={config.max_time}s, "
        f"warmup={config.warmup}, warmup_iterations={config.warmup_iterations}, "
        f"timeout={timeout_str}"
    )

    func, graph_param_name, source_hash = _load_function_and_hash(config)
    dataset_paths = _load_dataset_paths(config)
    configs = _parse_configurations(config)

    _validate_function_parameters(func, graph_param_name, configs)

    checkpoint_file = str(BENCHMARK_DIR / "benchmark_checkpoint.jsonl")
    _recover_checkpoint_if_present(checkpoint_file, config.output)

    existing_results, configs_to_run, explicit_tasks = _prepare_run_scope(
        config, func.__name__, dataset_paths, configs
    )

    test_file = str(BENCHMARK_DIR / "temp_benchmark_test.py")
    temp_results_file = str(BENCHMARK_DIR / "temp_benchmark_results.json")
    conftest_file = str(BENCHMARK_DIR / "conftest.py")
    timeout_log_file = str(BENCHMARK_DIR / "timeout_log.json")

    test_file, nodes_total = _generate_test_file(
        config,
        func.__name__,
        graph_param_name,
        dataset_paths,
        configs_to_run,
        explicit_tasks,
        test_file,
        source_hash,
    )
    print(f"Test file generated at {test_file}")

    _generate_conftest(config, conftest_file, nodes_total)

    _run_and_persist_results(
        config,
        test_file,
        conftest_file,
        temp_results_file,
        timeout_log_file,
        checkpoint_file,
        existing_results,
        func.__name__,
        source_hash,
    )


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
