"""Benchmark ingestion, population, and query throughput for the graphDB module."""

import platform
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

from pyrigi.graphDB import GraphStoreService, QueryFilter
from pyrigi.graphDB.models import any_of

G6_DIR = Path("pyrigi/graphDB/outputs/g6")
RIGIDITY_COLUMNS = ("rigidity", "min_rigidity", "global_rigidity")


def _row(label: str, n: int, elapsed: float, unit: str = "graphs/s") -> None:
    rate = n / elapsed if elapsed > 0 else float("inf")
    print(f"  {label:<40}  {n:>7,}  {elapsed:>7.2f}s  {rate:>9,.0f} {unit}")


def main() -> None:
    print("System")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  SQLite  : {sqlite3.sqlite_version}")
    print(f"  CPU     : {platform.processor() or platform.machine()}")
    print()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as tmp:
        db_path = tmp.name

    print(f"{'Phase':<42}  {'N':>7}  {'Time':>8}  {'Throughput':>14}")
    print("  " + "-" * 76)

    # Phase 1: ingestion (structural columns only)
    with GraphStoreService(db_path) as store:
        t0 = time.perf_counter()
        stats = store.ingest(str(G6_DIR))
        elapsed = time.perf_counter() - t0
        _row("Ingestion (structural)", stats.inserted, elapsed)

    # Phase 2: population of each rigidity column
    with GraphStoreService(db_path) as store:
        for col in RIGIDITY_COLUMNS:
            t0 = time.perf_counter()
            ps = store.populate_column(col)
            elapsed = time.perf_counter() - t0
            _row(f"Population ({col})", ps.processed, elapsed)

    # Phase 3: representative queries
    with GraphStoreService(db_path) as store:
        queries = [
            (
                "Query: num_vertices = 7 (simple filter)",
                lambda s: s.fetch(filters=[QueryFilter("num_vertices", "=", 7)]),
            ),
            (
                "Query: rigidity = 2 (-1 sentinel strategy)",
                lambda s: s.fetch(filters=[QueryFilter("rigidity", "=", 2)]),
            ),
            (
                "Query: min_rigidity = 2 (encoding strategy)",
                lambda s: s.fetch(filters=[QueryFilter("min_rigidity", "=", 2)]),
            ),
            (
                "Query: OR on num_vertices + ORDER BY",
                lambda s: s.fetch(
                    expr=any_of(
                        QueryFilter("num_vertices", "=", 6),
                        QueryFilter("num_vertices", "=", 7),
                    ),
                    order_by="num_edges",
                ),
            ),
        ]
        for label, fn in queries:
            t0 = time.perf_counter()
            rows = fn(store)
            elapsed = time.perf_counter() - t0
            _row(label, len(rows), elapsed, unit="rows/s")


if __name__ == "__main__":
    main()
