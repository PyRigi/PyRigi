import os
import subprocess
import sys


def run_pytest_benchmark(
    test_file: str,
    temp_output_file: str = "temp_benchmark_results.json",
    min_rounds: int = 5,
    max_time: float = 0.05,
    warmup: str = "off",
    warmup_iterations: int = 1,
    timeout: float = None,
    timeout_log_file: str = None,
    checkpoint_file: str = None,
):
    """
    Run pytest on the generated test file with benchmark options.

    Args:
        test_file: Path to generated test file.
        temp_output_file: Temporary output file path (will be merged later).
        min_rounds: Minimum number of benchmark rounds per test case.
                    Lower values are fine for complexity analysis.
                    Default: 5.
        max_time: Maximum time (seconds) for pytest-benchmark's adaptive
                  round loop per test case.
        warmup: Warmup mode passed to pytest-benchmark.
                Choices: 'auto', 'on', 'off'.  Default: 'off'.
        warmup_iterations: Number of warmup rounds when warmup is 'on' or 'auto'.
                           Ignored when warmup='off'.  Default: 1.
        timeout: Per-test-case timeout in seconds. None to disable.
        timeout_log_file: Path where the conftest hook will write timeout data.
        checkpoint_file: Path to the JSONL checkpoint file. If provided, each
                         completed test writes its result there immediately.

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
        f"--benchmark-max-time={max_time}",
        "-v",
    ]

    env = os.environ.copy()
    if timeout_log_file:
        env["BENCHMARK_TIMEOUT_LOG"] = timeout_log_file
    if checkpoint_file:
        env["BENCHMARK_CHECKPOINT_FILE"] = checkpoint_file

    print(f"Running benchmark command: {' '.join(cmd)}")
    if timeout is not None:
        print(f"Per-test timeout: {timeout}s")

    try:
        subprocess.run(cmd, check=True, env=env)
        return temp_output_file

    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmarks: {e}")
        raise
