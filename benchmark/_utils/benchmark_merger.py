import json
import os
import shutil
import datetime
import hashlib
from typing import List, Dict, Any, Tuple, Set


def compute_benchmark_key(
    func_name: str, config: Dict[str, Any], graph_filename: str, graph_idx: int
) -> str:
    """
    Generate unique key for a benchmark run.
    Key structure: function::config_hash::graph_file::graph_index
    """
    # Normalize config by sorting keys
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    return f"{func_name}::{config_hash}::{graph_filename}::{graph_idx}"


def load_existing_results(path: str) -> Dict[str, Any]:
    """Load existing benchmark results from JSON file."""
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(
            f"Warning: Could not parse existing results file {path}. Treating as empty."
        )
        return {}


def get_existing_keys(results: Dict[str, Any]) -> Set[str]:
    """Extract set of unique keys from existing results."""
    keys = set()
    for b in results.get("benchmarks", []):
        # Extract metadata
        func = b.get("function", "unknown")

        # Cases where function name might not be present
        if func == "unknown" and "name" in b:
            pass

        config = b.get("params", {}).get("config", {})

        graph_info = b.get("params", {}).get("graph_info", {})
        graph_file = graph_info.get("file_name", "unknown")
        graph_idx = graph_info.get("graph_idx", 0)

        key = compute_benchmark_key(func, config, graph_file, graph_idx)
        keys.add(key)

    return keys


def filter_existing_combinations(
    configs: List[Dict[str, Any]],
    graph_infos: List[Dict[str, Any]],
    existing_results: Dict[str, Any],
    target_function: str,
) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], int]:
    """
    Filter out combinations that already exist in results.
    Returns a list of (config, graph_info) tuples that need to be run.
    """
    existing_keys = get_existing_keys(existing_results)

    missing_combinations = []

    for config in configs:
        for graph_info in graph_infos:
            key = compute_benchmark_key(
                target_function,
                config,
                graph_info.get("file_name"),
                graph_info.get("graph_idx"),
            )

            if key not in existing_keys:
                missing_combinations.append((config, graph_info))

    initial_total = len(configs) * len(graph_infos)
    final_total = len(missing_combinations)
    num_skipped = initial_total - final_total

    return missing_combinations, num_skipped


def create_backup(path: str) -> str:
    """Create timestamped backup of results file."""
    if not os.path.exists(path):
        return ""

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.backup_{timestamp}"
    shutil.copy2(path, backup_path)
    return backup_path


def merge_results(
    existing_results: Dict[str, Any],
    new_results: Dict[str, Any],
    target_function: str,
    force_rerun: bool,
) -> Dict[str, Any]:
    """
    Merge new results into existing results.
    """
    merged = existing_results.copy()

    if "benchmarks" not in merged:
        merged["benchmarks"] = []

    # If force_rerun, remove old benchmarks for this function
    if force_rerun:
        print(f"Force rerun: removing existing results for '{target_function}'")
        merged["benchmarks"] = [
            b for b in merged["benchmarks"] if b.get("function") != target_function
        ]

    # Append new benchmarks
    existing_keys = get_existing_keys(merged)

    count_added = 0
    for b in new_results.get("benchmarks", []):
        config = b.get("params", {}).get("config", {})
        graph_info = b.get("params", {}).get("graph_info", {})
        key = compute_benchmark_key(
            b.get("function"),
            config,
            graph_info.get("file_name"),
            graph_info.get("graph_idx"),
        )

        if key not in existing_keys:
            merged["benchmarks"].append(b)
            existing_keys.add(key)
            count_added += 1

    # Update metadata
    if "metadata" not in merged:
        merged["metadata"] = {}

    merged["metadata"]["last_updated"] = datetime.datetime.now().isoformat()

    return merged
