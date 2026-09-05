"""
Internal helper modules for the benchmarking framework.

Each module owns one stage of the pipeline:
  cli, config_loader, param_parser  - resolve CLI and YAML input into a RunConfig.
  function_loader, dataset_loader   - load the target function and graph files.
  test_generator_mult               - emit the temporary pytest file and conftest.
  runner                            - invoke pytest-benchmark in a subprocess.
  checkpoint, benchmark_merger      - persist, deduplicate and merge results.
  models                            - the RunConfig dataclass shared by all stages.
"""
