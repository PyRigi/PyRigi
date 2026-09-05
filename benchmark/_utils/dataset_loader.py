"""
Discovery and inspection of the graph6 datasets a benchmark runs against.

get_dataset_files - list the .g6 files in a dataset directory.
load_graph_infos  - read those files into per-graph metadata dicts.

Only metadata is loaded here, not the graphs themselves; the generated test file
re-reads each graph from disk at measurement time so that graph construction is
never included in the timing.
"""

from pathlib import Path
from typing import List, Optional


def get_dataset_files(directory: str) -> List[str]:
    """
    List the .g6 files in a dataset directory.

    Args:
        directory: Path to the dataset directory.

    Returns:
        Sorted list of absolute paths to the .g6 files found.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path exists but is not a directory.
    """
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    return sorted([str(p.absolute()) for p in path.glob("*.g6")])


def load_graph_infos(
    dataset_paths: List[str],
    func_name: str,
    limit_per_file: Optional[int] = None,
) -> List[dict]:
    """
    Read all .g6 files and return a flat list of graph_info dicts.

    Each dict contains: file_idx, graph_idx, file_path, file_name,
    num_nodes, num_edges, function_name.

    Args:
        dataset_paths: Absolute paths to .g6 dataset files.
        func_name: Name of the benchmark function (stored in each dict).
        limit_per_file: Max number of graphs to load per file. None loads all.

    Returns:
        Flat list of graph_info dicts, sorted by vertex count ascending.
    """
    # Imported here rather than at module scope: get_dataset_files is used
    # during argument validation, which should not pay for the networkx import.
    import networkx as nx

    graph_infos = []

    for file_idx, graph_path in enumerate(dataset_paths):
        graph_filename = Path(graph_path).stem
        graph_data = nx.read_graph6(graph_path)

        if hasattr(graph_data, "__iter__") and not isinstance(graph_data, nx.Graph):
            graphs = list(graph_data)
        else:
            graphs = [graph_data]

        for graph_idx, graph in enumerate(graphs[:limit_per_file]):
            graph_infos.append(
                {
                    "file_idx": file_idx,
                    "graph_idx": graph_idx,
                    "file_path": graph_path,
                    "file_name": graph_filename,
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "function_name": func_name,
                }
            )

    # Sort graphs by size (ascending) to run smaller graphs first.
    graph_infos.sort(key=lambda g: g["num_nodes"])

    return graph_infos
