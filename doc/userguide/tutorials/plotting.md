---
jupytext:
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

# Plotting

+++

This notebook can be downloaded {download}`here <../../notebooks/plotting.ipynb>`.

```{code-cell} ipython3
# The import will work if the package was installed using pip.
import pyrigi.frameworkDB as frameworks
import pyrigi.graphDB as graphs
from pyrigi import Graph, Framework
```

Methods {meth}`.Graph.plot` and {meth}`.Framework.plot` offer various plotting options.
The default behaviour is the following:

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
G.plot()
```

```{code-cell} ipython3
F = Framework(G, {0:(0,0), 1:(1,1), 2:(3,1), 3:(2,0)})
F.plot()
```

##  Graph layouts
By default, a placement is generated using {func}`~networkx.drawing.layout.spring_layout`.

```{code-cell} ipython3
G = graphs.ThreePrism()
G.plot()
```

Other options use {func}`~networkx.drawing.layout.random_layout`, {func}`~networkx.drawing.layout.circular_layout` or {func}`~networkx.drawing.layout.planar_layout`:

```{code-cell} ipython3
G.plot(layout="random")
G.plot(layout="circular")
G.plot(layout="planar")
```

One can also specify a placement of the vertices explicitly:

```{code-cell} ipython3
G.plot(placement={0: (0, 0), 1: (0, 2), 2: (2, 1), 3: (6, 0), 4: (6, 2), 5: (4, 1)})
```

## Canvas options

The size of the canvas can be specified.

```{code-cell} ipython3
S = frameworks.Square()
S.plot(canvas_width=2)
```

```{code-cell} ipython3
S.plot(canvas_height=2)
```

```{code-cell} ipython3
S.plot(canvas_width=2, canvas_height=2)
```

Also the aspect ratio:

```{code-cell} ipython3
S.plot(aspect_ratio=0.4)
```

## Formatting
There are various options to format a plot.

Vertex color/size or label color/size can be changed.

```{code-cell} ipython3
G = Graph([[0,1]])
formatting = {
    "placement" : {0:[0,0], 1:[1,0]},
    "canvas_height" : 1,
}
G.plot(vertex_labels=False, vertex_color='green', **formatting)
G.plot(vertex_size=1500, font_size=30, font_color='#FFFFFF', **formatting)
```

There are various styles of vertices and edges.

```{code-cell} ipython3
formatting["vertex_labels"] = False
G.plot(vertex_shape='s', edge_style='-', **formatting)
G.plot(vertex_shape='o', edge_style='--', **formatting)
G.plot(vertex_shape='^', edge_style='-.', **formatting)
G.plot(vertex_shape='>', edge_style=':', **formatting)
G.plot(vertex_shape='v', edge_style='solid', **formatting)
G.plot(vertex_shape='<', edge_style='dashed', **formatting)
G.plot(vertex_shape='d', edge_style='dashdot', **formatting)
G.plot(vertex_shape='p', edge_style='dotted', **formatting)
G.plot(vertex_shape='h', edge_width=3, **formatting)
G.plot(vertex_shape='8', edge_width=5, **formatting)
```

## Edge coloring

The color of all edges can be changed.

```{code-cell} ipython3
P = graphs.Path(6)
formatting = {
    "placement" : {v:[v,0] for v in P.vertex_list()},
    "canvas_height" : 2,
    "edge_width" : 5,
}
P.plot(edge_color='red', **formatting)
```

If a partition of the edges is specified, then each part is colored differently.

```{code-cell} ipython3
P.plot(edge_color=[[[0,1],[2,3]], [[1,2]], [[5,4],[4,3]]], **formatting)
```

If the partition is incomplete, the missing edges are black.

```{code-cell} ipython3
P.plot(edge_color=[[[0,1],[2,3]], [[5,4],[4,3]]], **formatting)
```

Visually distinct colors are generated using the package [`distinctipy`](https://pypi.org/project/distinctipy/).

```{code-cell} ipython3
P30 = graphs.Path(30)
P30.plot(vertex_size=15,
        vertex_labels=False,
        edge_color=[[e] for e in P30.edge_list()],
        edge_width=3
       )
```

Another possibility is to provide a dictionary assigning to a color a list of edges. Missing edges are again black.

```{code-cell} ipython3
P.plot(edge_color={
    "yellow" : [[0,1],[2,3]],
    "#ABCDEF": [[5,4],[4,3]]},
       **formatting)
```

## Framework plotting

Currently, only plots of frameworks in the plane are implemented.

```{code-cell} ipython3
F = frameworks.Complete(9)
F.plot()
```

The same formatting options as for graphs are available for frameworks.

```{code-cell} ipython3
F = frameworks.Complete(9)
F.plot(vertex_labels=False,
       vertex_color='#A2B4C6',
       edge_style='dashed',
       edge_width=2,
       edge_color={"pink" : [[0,1],[3,6]], "lightgreen" : [[2,3],[3,5]]}
      )
```
