from pathlib import Path
from typing import List


def get_dataset_files(directory: str) -> List[str]:
    """
    Returns list of absolute paths to .g6 files in the directory.
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
    limit_per_file: int = 10,
) -> List[dict]:
    """
    Read all .g6 files and return a flat list of graph_info dicts.

    Each dict contains: file_idx, graph_idx, file_path, file_name,
    num_nodes, num_edges, function_name.

    Args:
        dataset_paths: Absolute paths to .g6 dataset files.
        func_name: Name of the benchmark function (stored in each dict).
        limit_per_file: Max number of graphs to load per file.

    Returns:
        Flat list of graph_info dicts.
    """
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
