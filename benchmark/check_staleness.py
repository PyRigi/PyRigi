#!/usr/bin/env python3
import json
import sys
import os
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark._utils import function_loader  # noqa: E402


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Error: Results file not found at {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def check_staleness(results_path: str):
    print(f"Checking staleness for results in {results_path}...")
    try:
        data = load_results(results_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        print("No benchmarks found.")
        return

    unique_funcs = set(b.get("function") for b in benchmarks if b.get("function"))
    print(f"Found {len(unique_funcs)} unique functions in results.")

    live_hashes = {}

    print(f"{'Function':<30} | {'Module':<40} | {'Status':<15} | {'Details'}")
    print("-" * 110)

    stale_entries = 0
    total_entries = 0
    processed_funcs = set()

    for b in benchmarks:
        total_entries += 1
        func_name = b.get("function", "unknown")
        module_path = b.get("module_path")
        stored_hash = b.get("source_hash")

        hash_msg = ""
        is_changed = False

        if module_path and func_name:
            key = (func_name, module_path)

            # Compute current hash only once per function/module pair
            if key not in live_hashes:
                try:
                    target = module_path
                    if ":" not in target:
                        target = f"{target}:{func_name}"

                    func_obj, _ = function_loader.load_function_and_detect_param(target)
                    current_hash = function_loader.get_function_hash(func_obj)
                    live_hashes[key] = current_hash
                except Exception:
                    live_hashes[key] = None

            current_hash = live_hashes[key]

            if current_hash:
                if not stored_hash:
                    hash_msg = "No stored hash"
                elif current_hash != stored_hash:
                    hash_msg = "Code Changed"
                    is_changed = True
                else:
                    hash_msg = "Verified"
            else:
                hash_msg = "Load Failed"
        else:
            hash_msg = "No Path Info"

        if is_changed:
            stale_entries += 1

            # Print only once per unique function to avoid spam
            dict_key = f"{func_name}::{module_path}"
            if dict_key not in processed_funcs:
                status = "STALE"
                print(
                    f"{func_name:<30} | {str(module_path)[:40]:<40} | "
                    f"{status:<15} | {hash_msg}"
                )
                processed_funcs.add(dict_key)

    print("-" * 110)
    print(f"Total Benchmarks: {total_entries}")
    print(f"Stale Benchmarks: {stale_entries} (candidates for re-running)")
    if stale_entries == 0:
        print("All benchmarks are fresh!")


def main():
    parser = argparse.ArgumentParser(description="Check benchmark staleness.")
    parser.add_argument(
        "--results",
        default="benchmark/benchmark_results.json",
        help="Path to results file",
    )

    args = parser.parse_args()
    check_staleness(args.results)


if __name__ == "__main__":
    main()
