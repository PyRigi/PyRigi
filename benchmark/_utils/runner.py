import subprocess
import sys
import json


def run_pytest_benchmark(test_file: str, output_file: str):
    """
    Run pytest on the generated test file with benchmark options.
    Uses auto warmup for efficiency and post-processes JSON to remove bloat.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,median,iterations",
        f"--benchmark-json={output_file}",
        "--benchmark-disable-gc",
        "--benchmark-warmup=auto",
        "--benchmark-min-rounds=10",
        "-v",
    ]

    print(f"Running benchmark command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)

        # Post-process JSON to remove bloat
        print(f"Post-processing {output_file} to remove raw iteration data...")
        with open(output_file, "r") as f:
            data = json.load(f)

        # Remove 'data' field from stats in each benchmark
        benchmarks_cleaned = 0
        for benchmark in data.get("benchmarks", []):
            if "stats" in benchmark and "data" in benchmark["stats"]:
                del benchmark["stats"]["data"]
                benchmarks_cleaned += 1

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Cleaned {benchmarks_cleaned} benchmark(s) - removed raw iteration data")

    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmarks: {e}")
        raise
    except Exception as e:
        print(f"Warning: JSON post-processing failed: {e}")
        # Don't raise - the benchmark data is still valid
