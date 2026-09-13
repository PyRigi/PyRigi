(benchmarking)=
# Benchmarking

PyRigi offers several algorithms for the same decision problem, and the fastest one
often depends on the input. The benchmarking framework in the `benchmark` folder
measures any Python function that accepts a NetworkX graph, over a dataset of
graph6 files, across a parameter space you declare. Each measurement is taken by
`pytest-benchmark` in a separate subprocess, so any function reachable by a file
path can be benchmarked without changing framework code.

The framework is a development tool: it lives in the repository but is not part of
the installed package, so all commands below are run from a clone, with the Poetry
environment [activated](#dependencies-poetry).

Working with it has three stages:

| Stage | What you do | Result |
|-------|-------------|--------|
| 1. Prepare | Collect the graphs to measure against into a directory of `.g6` files | A dataset |
| 2. Run | Call `benchmarkapp.py` with a target function and that dataset | A results JSON file |
| 3. Analyse | Run the analysis notebook on the results | Summary tables and figures |

## Preparing a dataset

No dataset ships with the repository, so the first step is to assemble one from
the graphs you want to measure against.

A dataset is simply a directory of [graph6](https://users.cecs.anu.edu.au/~bdm/data/formats.txt)
(`.g6`) files. Each file holds one graph per line, and a file containing several
graphs is expanded into one benchmark case per graph. By convention one file holds
all graphs of a given vertex count, which keeps the results easy to group by size:

```
benchmark/graph_store/my_dataset/
├── my_dataset_10.g6
├── my_dataset_11.g6
└── my_dataset_12.g6
```

Any source of graphs works. For instance, to write ten random graphs for each
vertex count from 10 to 20 using NetworkX:

```python
from pathlib import Path
import networkx as nx

out = Path("benchmark/graph_store/my_dataset")
out.mkdir(parents=True, exist_ok=True)

for n in range(10, 21):
    graphs = [nx.gnm_random_graph(n, 2 * n - 3, seed=s) for s in range(10)]
    nx.write_graph6(graphs[0], out / f"my_dataset_{n}.g6", header=False)
    with open(out / f"my_dataset_{n}.g6", "ab") as f:
        for graph in graphs[1:]:
            nx.write_graph6(graph, f, header=False)
```

Use a fixed seed so that a rerun measures the same graphs.

## Running a benchmark

The entry point is `benchmark/benchmarkapp.py`. The two required inputs are the
target function, given as `path/to/file.py:function_name`, and the dataset
directory. Parameters to sweep are listed as `name=val1,val2`, and the framework
benchmarks the Cartesian product of the listed values:

```
python benchmark/benchmarkapp.py pyrigi/graph/_rigidity/generic.py:is_min_rigid \
  --dataset benchmark/graph_store/my_dataset \
  --params dim=2 algorithm=sparsity,randomized
```

The `target` and `--dataset` paths are resolved against the current working
directory, so they must match where you run the command from; the example above
assumes the repository root. A relative `--output` is the exception, as it is
always placed in the `benchmark` folder.

The most useful options are:

| Option | Effect |
|--------|--------|
| `--min-rounds` | Minimum measurement rounds per case (default `5`) |
| `--max-time` | Budget in seconds for the adaptive round loop per case (default `0.05`) |
| `--warmup` | Warmup mode, `auto`, `on` or `off` (default `off`) |
| `--timeout` | Per-case timeout in seconds (default: none) |
| `--timeout-threshold` | Early-stop threshold between `0.0` and `1.0` (default: none) |
| `--force-rerun` | Measure again instead of skipping cases that already have results |

Run `python benchmark/benchmarkapp.py --help` for the complete list.

Repeated settings are better kept in a YAML configuration file, passed with
`--config`. The file `benchmark/benchmark_config.example.yaml` is a documented
template that explains every key inline; copy it and point its `dataset` key at
your own directory:

```
python benchmark/benchmarkapp.py --config benchmark/my_config.yaml
```

Command-line flags override values from the configuration file, so you can reuse
one file and vary a single setting on the fly.

Results are merged into the output file rather than replacing it, and a case that
already has a result is skipped. This makes it cheap to extend an existing study
with another algorithm or a few larger graphs.

## Timeouts and early stopping

Rigidity computations on larger graphs can take a very long time, so a run can be
bounded in two ways.

`--timeout` sets a per-case limit. A case that exceeds it is abandoned and recorded
as a timeout rather than a measurement.

`--timeout-threshold` adds early stopping. When the fraction of timed-out cases at
one vertex count reaches the threshold, that configuration is not attempted at any
larger size. The threshold is tracked per configuration, so a slow algorithm being
stopped does not affect the others in the same run. For example,
`--timeout 3 --timeout-threshold 0.75` abandons any case taking longer than three
seconds and stops a configuration once three quarters of the cases at some size
have timed out.

Timeouts are summarised on the console and saved to `timeout_results.json` next to
the results file.

:::{note}
Both features rely on `signal.SIGALRM`, which exists only on POSIX systems. On
native Windows the benchmark still runs, but the timeout is not enforced (a
`RuntimeWarning` is emitted) and early stopping never triggers, so slow cases run
to completion. Running under WSL gives the documented behaviour.
:::

## Resuming and checking staleness

Every completed case is appended to `benchmark/benchmark_checkpoint.jsonl` as soon
as it finishes. If a session is interrupted, the next run merges that checkpoint
into the results file and continues, so no completed measurement is lost and none
is repeated.

Because results accumulate over time, they can outlive the code that produced them.
Each entry records a hash of the benchmarked function's source, and

```
python benchmark/check_staleness.py --results benchmark/benchmark_results.json
```

reports which stored results no longer match the current implementation. The hash
is computed from the syntax tree with comments and docstrings removed, so
reformatting or re-documenting a function does not mark its results stale; only a
change to the code itself does.

## Analysing the results

The analysis stage is a Jupyter notebook stored as
[jupytext](https://jupytext.readthedocs.io/) MyST markdown in
`benchmark/analysis.md`. Generate the runnable notebook with

```
jupytext --to notebook benchmark/analysis.md
```

then open `benchmark/analysis.ipynb`, select the Poetry environment as the kernel,
and choose *Restart & Run All*. The notebook is driven by a single configuration
cell (§ 1) holding the results path, the filters and the plot settings; the
remaining sections are meant to be run unmodified.

It produces the following views:

| View | What it shows | Statistical basis |
|------|---------------|-------------------|
| Scaling plot | Mean time against graph size for each configuration | 95% confidence band (Student-t or bootstrap) with a power-law fit |
| Comparison bars | Mean time at selected sizes, side by side | 95% confidence intervals and a Welch t-test |
| Relative-performance heatmap | Slowdown of each configuration against the fastest at each size | Ratio of means |
| Stability heatmap | Measurement noise per configuration and size | Coefficient of variation |
| Instance box plots | Spread of per-instance times at each size | Quartiles and interquartile range |
| Timeout-rate plot | Fraction of timed-out cases per configuration and size | Count over attempted cases |

The confidence bands and the t-test are computed across the graph instances at a
given size, which is what makes a comparison between two algorithms meaningful
rather than a comparison of two single timings.

## Files written by a run

| File | Contents |
|------|----------|
| `benchmark_results.json` | The measurements, merged across runs. The name follows `--output` |
| `timeout_results.json` | Per-run timeout and early-stop summary |
| `benchmark_checkpoint.jsonl` | Crash-safe log of completed cases, merged and removed on the next run |
| `early_stop_state.json` | Configurations stopped early, so a resumed run skips them |

The first two are placed next to the results file, wherever `--output` points; the
last two always live in the `benchmark` folder.

A run also generates `temp_benchmark_test.py` and `conftest.py` in the `benchmark`
folder. These carry out the measurements and are deleted when the run finishes;
they are not meant to be edited.
