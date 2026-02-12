from typing import List, Dict, Any


def generate_benchmark_test_file(
    func_path: str,
    func_name: str,
    graph_param_name: str,
    dataset_paths: List[str],
    configurations: List[Dict[str, Any]],
    output_path: str = "temp_benchmark_test.py",
) -> str:
    """
    Generate a pytest file using parametrization to capture config values in JSON output.
    Only benchmarks the first graph from each .g6 file.
    """

    file_abs_path = func_path.split(":")[0]

    # Build parameter combinations: (config_dict, graph_path)
    param_combinations = []
    for config in configurations:
        for graph_path in dataset_paths:
            param_combinations.append((config, graph_path))

    # Build complete test file with parametrization
    content = f"""import pytest
import networkx as nx
import sys
import importlib.util

# Dynamic Import
try:
    spec = importlib.util.spec_from_file_location("target_module", r"{file_abs_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["target_module"] = module
    spec.loader.exec_module(module)
    target_func = getattr(module, "{func_name}")
except Exception as e:
    raise RuntimeError(f"Failed to import function inside generated test: {{e}}")

# Parametrized Benchmark Test
@pytest.mark.parametrize("config,graph_path", {repr(param_combinations)})
def test_benchmark(benchmark, config, graph_path):
    \"\"\"
    Benchmark the target function with given config and graph.
    pytest-benchmark will automatically capture 'config' and 'graph_path' in JSON output.
    \"\"\"
    # Load graph
    graph_data = nx.read_graph6(graph_path)


    # Handle both single graph and multiple graphs in file
    if hasattr(graph_data, '__iter__') and not isinstance(graph_data, nx.Graph):
        graphs = list(graph_data)
        graph = graphs[0]  # Take only the first graph
    else:
        graph = graph_data


    # Prepare kwargs with graph parameter
    kwargs = config.copy()
    kwargs['{graph_param_name}'] = graph


    # Run benchmark
    benchmark(target_func, **kwargs)
"""

    with open(output_path, "w") as f:
        f.write(content)

    return output_path
