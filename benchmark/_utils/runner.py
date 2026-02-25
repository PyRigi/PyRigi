import subprocess
import sys


def run_pytest_benchmark(
    test_file: str,
    temp_output_file: str = "temp_benchmark_results.json",
    min_rounds: int = 5,
    warmup: str = "off",
    warmup_iterations: int = 1,
):
    """
    Run pytest on the generated test file with benchmark options.

    Args:
        test_file: Path to generated test file.
        temp_output_file: Temporary output file path (will be merged later).
        min_rounds: Minimum number of benchmark rounds per test case.
                    Lower values are fine for complexity analysis.
                    Default: 5.
        warmup: Warmup mode passed to pytest-benchmark.
                Choices: 'auto', 'on', 'off'.  Default: 'off'.
        warmup_iterations: Number of warmup rounds when warmup is 'on' or 'auto'.
                           Ignored when warmup='off'.  Default: 1.

    Returns:
        Path to temporary results file.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,median,iterations",
        f"--benchmark-json={temp_output_file}",
        "--benchmark-disable-gc",
        f"--benchmark-warmup={warmup}",
        f"--benchmark-warmup-iterations={warmup_iterations}",
        f"--benchmark-min-rounds={min_rounds}",
        "-v",
    ]

    print(f"Running benchmark command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return temp_output_file

    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmarks: {e}")
        raise
