from pyrigi.graph import Graph

import os
import networkx as nx


def read_graph_from_file(filename):
    with open(filename) as f:
        n, m = [int(x) for x in next(f).split()]
        g = Graph()
        for i in range(n):
            g.add_vertex(i)
        for i in range(m):
            v1, v2 = [int(x) for x in next(f).split()]
            g.add_edge(v1, v2)
        return g


def convert_and_save_graphs(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):  # Adjust the file extension as needed
            file_path = os.path.join(input_folder, filename)
            graph = read_graph_from_file(file_path)
            if (filename == "sparse_1_1.txt"):
                print(graph.edges)
            if isinstance(graph, Graph):
                print(len(graph.edges))
                sparse6_bytes = nx.to_sparse6_bytes(graph)
                output_file_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}.s6"
                )

                with open(output_file_path, "wb") as f:
                    f.write(sparse6_bytes)
                    print(f"Converted {filename} to sparse6 format.")
            else:
                print(f"File {filename} is not a valid MyGraph instance.")


input_folder = "input_graphs"
output_folder = "input_graphs"

convert_and_save_graphs(input_folder, output_folder)

print("Conversion complete.")