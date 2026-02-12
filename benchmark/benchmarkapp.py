import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

from _utils import (  # noqa: E402
    function_loader,
    param_parser,
    dataset_loader,
    runner,
    config_loader,
    test_generator_mult,
)


def load_and_merge_config(args):
    """Load config from file and merge with CLI arguments."""
    if args.config:
        print(f"Loading configuration from {args.config}...")
        try:
            config = config_loader.load_config(args.config)
            config_loader.validate_config(config)
            merged_config = config_loader.merge_with_cli(config, args)
            return merged_config, True
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    return {}, False


def validate_cli_args(args):
    """Validate CLI arguments when no config file is provided."""
    target = args.target
    dataset = args.dataset
    output = args.output if args.output else "benchmark_results.json"
    params = args.params if args.params else []

    if not target:
        sys.stderr.write("error: target is required when not using --config\n")
        sys.exit(1)
    if not dataset:
        sys.stderr.write("error: --dataset is required when not using --config\n")
        sys.exit(1)

    return target, dataset, output, params


def main():
    parser = argparse.ArgumentParser(description="PyRigi Centralized Benchmarking Tool")

    # Config file support
    parser.add_argument("--config", help="Path to YAML config file (optional)")

    parser.add_argument(
        "target",
        nargs="?",  # Make optional if config is provided
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

    args = parser.parse_args()

    # Load and merge configuration
    merged_config, loaded_from_config = load_and_merge_config(args)

    if loaded_from_config:
        target = merged_config.get("target")
        dataset = merged_config.get("dataset")
        output = merged_config.get("output", "benchmark_results.json")
        params = merged_config.get("cli_params", [])
    else:
        target, dataset, output, params = validate_cli_args(args)

    # 1. Load Function & Detect Graph Parameter
    print(f"Loading function from {target}...")
    try:
        func, graph_param_name = function_loader.load_function_and_detect_param(target)
        print(
            f"Found function '{func.__name__}' with graph parameter '{graph_param_name}'"
        )
    except Exception as e:
        print(f"Error loading function: {e}")
        sys.exit(1)

    # 2. Load Dataset
    print(f"Loading dataset from {dataset}...")
    try:
        dataset_paths = dataset_loader.get_dataset_files(dataset)
        print(f"Found {len(dataset_paths)} graph files.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if not dataset_paths:
        print("No .g6 files found in dataset directory.")
        sys.exit(1)

    # 3. Parse Parameters
    print("Parsing parameters...")

    # Use merged configurations from config file if available
    if loaded_from_config and "configurations" in merged_config:
        configs = merged_config["configurations"]
        print(
            f"Loaded {len(configs)} configuration(s) from config file "
            f"(cartesian + explicit)."
        )
    else:
        # Fallback to CLI parameter parsing (cartesian product only)
        raw_params = param_parser.parse_cli_strings(params)
        configs = param_parser.build_cartesian_product(raw_params)
        print(
            f"Generated {len(configs)} configuration(s) from CLI "
            f"using Cartesian product strategy."
        )

    # 3.5. Validate Parameters Against Function Signature
    if configs and configs[0]:
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

    # 4. Generate Test File
    print("Generating temporary test file...")
    try:
        test_file = test_generator_mult.generate_benchmark_test_file(
            func_path=target,
            func_name=func.__name__,
            graph_param_name=graph_param_name,
            dataset_paths=dataset_paths,
            configurations=configs,
        )
        print(f"Test file generated at {test_file}")
    except Exception as e:
        print(f"Error generating test file: {e}")
        sys.exit(1)

    # 5. Run Benchmarks
    print("Running benchmarks...")
    try:
        runner.run_pytest_benchmark(test_file, output)
        print(f"Benchmarks completed. Results saved to {output}")
    except Exception as e:
        print(f"Benchmark run failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_file):
            print("Cleaning up temporary test file...")
            os.remove(test_file)


if __name__ == "__main__":
    main()
