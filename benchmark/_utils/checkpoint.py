"""
Crash-safe per-test checkpointing for the benchmark pipeline.

write_entry         — append one JSON line to the JSONL checkpoint after each test.
drain_into_results  — on startup: read checkpoint, deduplicate, merge atomically, delete.
clear               — delete the checkpoint file.
exists              — return True if a non-empty checkpoint file is present.

Crash safety:
  write_entry uses append mode — only the last line can be malformed on crash.
  drain_into_results writes to .tmp then os.replace() (atomic on POSIX) to
  ensure the main results JSON is never left half-written.
"""

import datetime
import json
import os
from typing import Any, Dict

from _utils.benchmark_merger import compute_benchmark_key, get_existing_keys


def write_entry(checkpoint_path: str, entry: Dict[str, Any]) -> None:
    """
    Append one benchmark result as a JSON line to checkpoint_path.

    Append mode: safe to call after every test; a mid-write crash corrupts
    only the last line — all preceding lines remain intact.
    """
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def drain_into_results(
    checkpoint_path: str, results_path: str, only_timeouts: bool = False
) -> int:
    """
    Merge checkpoint into results_path, then delete the checkpoint.

    Called on startup (crash recovery) or after a clean run.
    Malformed lines (partial crash writes) are silently skipped.
    Deduplicates via compute_benchmark_key — no entry is written twice.
    Write is atomic: results go to <results_path>.tmp, then os.replace().

    Args:
        only_timeouts: If True, merge only timeout markers and skip completed
            entries (which already come from pytest-benchmark's JSON on a clean run).

    Returns number of new entries added.
    """
    if not exists(checkpoint_path):
        return 0

    # 1. Read checkpoint lines
    entries = []
    with open(checkpoint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Partial last line from crash — discard.

    if not entries:
        clear(checkpoint_path)
        return 0

    # 2. Load existing results (or start fresh)
    existing: Dict[str, Any] = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: Could not parse {results_path}. "
                "Treating as empty for merge."
            )

    existing.setdefault("benchmarks", [])
    existing.setdefault("metadata", {})

    # 3. Deduplicate and merge
    known_keys = get_existing_keys(existing)
    count_added = 0

    for entry in entries:
        if only_timeouts and not entry.get("timed_out"):
            continue  # clean-run drain: completed entries are already in results
        params = entry.get("params", {})
        config = params.get("config", {})
        graph_info = params.get("graph_info", {})
        func = entry.get("function", "unknown")

        key = compute_benchmark_key(
            func,
            config,
            graph_info.get("file_name", "unknown"),
            graph_info.get("graph_idx", 0),
        )

        if key not in known_keys:
            existing["benchmarks"].append(entry)
            known_keys.add(key)
            count_added += 1

    existing["metadata"]["last_updated"] = datetime.datetime.now().isoformat()

    # 4. Atomic write
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp_path, results_path)

    # 5. Remove checkpoint now that it's safely merged
    clear(checkpoint_path)

    return count_added


def clear(checkpoint_path: str) -> None:
    """Delete checkpoint_path if it exists. OSErrors are suppressed."""
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError as e:
            print(f"Warning: Could not remove checkpoint file {checkpoint_path}: {e}")


def exists(checkpoint_path: str) -> bool:
    """
    Return True if checkpoint_path exists and is non-empty.
    Empty files are treated as absent to prevent runaway recovery loops.
    """
    return os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0
