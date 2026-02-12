from typing import List, Dict, Any
from pathlib import Path
import networkx as nx


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
    Pre-loads all graphs to handle multi-graph .g6 files correctly.
    """

    file_abs_path = func_path.split(":")[0]

    print("Pre-loading graphs from dataset files...")
    all_graphs = []

    for file_idx, graph_path in enumerate(dataset_paths):
        graph_filename = Path(graph_path).stem

        graph_data = nx.read_graph6(graph_path)

        if hasattr(graph_data, "__iter__") and not isinstance(graph_data, nx.Graph):
            graphs = list(graph_data)
        else:
            graphs = [graph_data]

        for graph_idx, graph in enumerate(graphs[:10]):
            all_graphs.append(
                {
                    "file_idx": file_idx,
                    "graph_idx": graph_idx,
                    "file_path": graph_path,
                    "file_name": graph_filename,
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                }
            )

    print(f"Loaded {len(all_graphs)} graphs from {len(dataset_paths)} files")

    # Parameter combinations: (config_dict, graph_metadata)
    param_combinations = []
    for config in configurations:
        for graph_info in all_graphs:
            param_combinations.append((config, graph_info))

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

# Parametrized Benchmark
@pytest.mark.parametrize("config,graph_info", {repr(param_combinations)})
def test_benchmark(benchmark, config, graph_info):
    \"\"\"
    Benchmark the target function with given config and graph.
    pytest-benchmark will automatically capture 'config' and 'graph_info' in JSON output.


    graph_info contains: file_path, file_idx, graph_idx, file_name, num_nodes, num_edges
    \"\"\"
    # Extract graph metadata
    file_path = graph_info['file_path']
    graph_idx = graph_info['graph_idx']


    # Load graph from file
    graph_data = nx.read_graph6(file_path)


    # Handle both single graph and multiple graphs in file
    if hasattr(graph_data, '__iter__') and not isinstance(graph_data, nx.Graph):
        graphs = list(graph_data)
        graph = graphs[graph_idx]
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
