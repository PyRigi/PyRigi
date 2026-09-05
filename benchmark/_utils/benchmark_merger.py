"""
Merging and deduplication of benchmark results.

compute_benchmark_key       - build the identity of a single measurement.
load_existing_results       - read a results JSON file, tolerating corruption.
get_existing_keys           - collect the keys already present in a results file.
filter_existing_combinations - work out which (config, graph) pairs still need running.
create_backup               - snapshot a results file before a force-rerun.
merge_results               - append new measurements without creating duplicates.

Identity:
  A measurement is identified by function, configuration, graph file and graph
  index (see compute_benchmark_key). Re-running the same combination therefore
  never appends a second entry, which is what makes accumulating results across
  many sessions safe.
"""

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
    Generate the unique key identifying a single benchmark measurement.

    Key structure: function::config_hash::graph_file::graph_index
    The config is hashed with its keys sorted, so two configurations that differ
    only in key order produce the same key.

    Args:
        func_name: Name of the benchmarked function.
        config: Parameter values the function was called with.
        graph_filename: Stem of the .g6 file the graph came from.
        graph_idx: Index of the graph within that file.

    Returns:
        The key string.
    """
    # Normalize config by sorting keys
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()[:8]

    return f"{func_name}::{config_hash}::{graph_filename}::{graph_idx}"


def load_existing_results(path: str) -> Dict[str, Any]:
    """
    Load existing benchmark results from a JSON file.

    A missing or unparseable file is reported and treated as empty rather than
    raising, so a corrupted results file cannot block a new run.

    Args:
        path: Path to the results JSON file.

    Returns:
        Parsed results, or an empty dict if the file is missing or invalid.
    """
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
    """
    Extract the set of benchmark keys already present in a results dict.

    Args:
        results: Parsed contents of a results JSON file.

    Returns:
        Set of keys as produced by compute_benchmark_key.
    """
    keys = set()
    for b in results.get("benchmarks", []):
        # Extract metadata
        func = b.get("function", "unknown")

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
    Determine which (config, graph) combinations have not been measured yet.

    Args:
        configs: Parameter configurations to benchmark.
        graph_infos: Graph metadata dicts from dataset_loader.load_graph_infos.
        existing_results: Parsed contents of the results file.
        target_function: Name of the function being benchmarked.

    Returns:
        Tuple of (missing_combinations, num_skipped), where missing_combinations
        is a list of (config, graph_info) pairs still to run and num_skipped is
        how many pairs were already present.
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
    """
    Create a timestamped copy of a results file before it is overwritten.

    Args:
        path: Path to the results file to back up.

    Returns:
        Path to the backup, or an empty string if the source did not exist.
    """
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
    Merge newly measured benchmarks into the existing results.

    Entries whose key is already present are dropped, so merging is idempotent.
    Under force_rerun the previous entries for target_function are removed first,
    which lets the fresh measurements take their place.

    Args:
        existing_results: Parsed contents of the results file.
        new_results: Freshly measured results from pytest-benchmark.
        target_function: Name of the function being benchmarked.
        force_rerun: Drop existing entries for target_function before merging.

    Returns:
        The merged results dict, with metadata.last_updated refreshed.
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
