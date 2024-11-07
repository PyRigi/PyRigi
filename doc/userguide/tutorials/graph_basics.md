---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Graph - basic manipulation

+++

This notebook illustrates the basic functionality of {class}`pyrigi.graph.Graph`.
It can be downloaded {download}`here <../../notebooks/graph_basics.ipynb>`.

```{code-cell} ipython3
# The import will work if the package was installed using pip.
from pyrigi import Graph
```

An easy way to construct a graph is to provide the list of its edges:

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
G
```

Edges and vertices can be added:

```{code-cell} ipython3
G.add_vertices([0, 2, 5, 7, 'a', 'b'])
G.add_edges([(0,7), (2,5)])
G
```

or removed:

```{code-cell} ipython3
G.delete_vertex('a')
G
```

```{code-cell} ipython3
G.delete_vertices([2, 7])
G
```

```{code-cell} ipython3
G.delete_edges([(0,1), (0,3)])
G
```

There are also other ways how to construct a graph:

```{code-cell} ipython3
import pyrigi.graphDB as graphs
graphs.Complete(4)
```

```{code-cell} ipython3
Graph.CompleteOnVertices(['a', 1, (1.2)])
```

```{code-cell} ipython3
from sympy import Matrix
Graph.from_adjacency_matrix(Matrix([[0,1,1], [1,0,0], [1,0,0]]))
```

```{code-cell} ipython3
Graph.from_vertices(range(4))
```

```{code-cell} ipython3
Graph.from_vertices_and_edges(range(6), [[i, (i+2) % 6] for i in range(6)])
```

A vertex of a graph can be of any hashable type, but it is recommended to have all of them of the same type, not as above. If all vertices have the same type, the vertex/edge set can be sorted when a list is required; otherwise, the order might differ:

```{code-cell} ipython3
G = Graph([[0, 7], [2, 5], [1, 2], [0, 1], [0, 3], [2, 3]])
print(G.vertex_list())
print(G.edge_list())
print(Graph.from_vertices(['a',1,(1,2)]).vertex_list())
print(Graph.from_vertices([1,'a',(1,2)]).vertex_list())
```

```{code-cell} ipython3

```
