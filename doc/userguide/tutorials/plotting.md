---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Plotting Graphs and Frameworks

+++

Methods {meth}`.Graph.plot` and {meth}`.Framework.plot` offer various plotting options.

```{code-cell} ipython3
# The import will work if:
#     - the tutorial is in the root folder of the package, or
#     - the package was installed using poetry,
#       see https://pyrigi.github.io/PyRigi/development/howto.html#dependencies, or
#     - the package is added to the sys.path using the following with the correct
#       path to the pyrigi root folder
import os, sys
sys.path.insert(0, os.path.abspath("../../.."))
import pyrigi.frameworkDB as frameworks
from pyrigi import Graph, Framework
```

## Default plotting options

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
F = Framework(G, {0:(0,0), 1:(1,1), 2:(3,1), 3:(2,0)})
G.plot()
F.plot()
```

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
G.plot(vertex_labels=False, vertex_color='#000000', vertex_size=400, vertex_shape='^', edge_color='#ff0000', edge_style=':')

H = Graph([(0,1)])
print(H)
H.plot(canvas_width=3,canvas_height=2)

I = frameworks.Path(29).graph()
I.plot(vertex_size=100, font_size=15, font_color='#ff00ff')
```

```{code-cell} ipython3
F = frameworks.Complete(9)
F.plot(vertex_labels=False, vertex_color='#123456', edge_style='dashed', edge_width=1.2)
E = frameworks.Path(6)
E.plot(font_color='y', font_size=40)
D = Framework(Graph([[0,1]]), {0:[0,5], 1:[0,0]})
```

```{code-cell} ipython3
for i in range(5):
    F = frameworks.Complete(i+1)
    F.plot(edge_width=i+0.1, canvas_width=2*(i+1), canvas_height=i+1, edge_color='yellow')
```

```{code-cell} ipython3
F1 = Framework(Graph([(0,1), (1,2), (2,3), (3,0)]), {0:(0,0), 1:(0,100), 2:(100,100), 3:(100,0)})
F1.plot(canvas_width=5, canvas_height=2, edge_color=[[(0,1),(1,2)],[(2,3),(3,0)]], edge_width=2)
F1.plot(edge_color=[[(0,1)],[(1,2)],[(2,3)],[(0,3)]])
F1.plot(edge_color={"pink":[(0,1),(1,2)],"maroon":[(2,3),(3,0),(1,2)]})
```

```{code-cell} ipython3
F = Framework(Graph([(0,1), (1,2), (2,3), (3,0)]), {0:(0,0), 1:(0,30), 2:(1,30), 3:(1,0)})
F.plot()
F.plot(aspect_ratio=0.25)
```

```{code-cell} ipython3
f = frameworks.Complete(5)
g = f.graph()
f.plot()
g.plot()
f.plot(canvas_width=8, canvas_height=6)
g.plot(canvas_width=8, canvas_height=6)
```

this is a markdown cell

```{code-cell} ipython3

```
