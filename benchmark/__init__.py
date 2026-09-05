"""
Benchmarking framework for PyRigi graph algorithms.

Measures the performance of any function that accepts a NetworkX graph, across
parametrised configurations and graph datasets, and stores aggregated timing
results as JSON for later analysis.

Entry points:
  benchmarkapp.py     - run a benchmark from a YAML config or CLI arguments.
  check_staleness.py  - report which stored results no longer match the code.
"""
