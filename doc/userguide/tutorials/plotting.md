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

```{code-cell} ipython3
import os, sys
sys.path.insert(0, os.path.abspath("../../.."))
import pyrigi.frameworkDB as frameworks
from pyrigi import Graph, Framework
```

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
H = Graph([(0,1)])

print(G)
# G.plot(vertex_labels=False, node_color='#000000')
H.plot()
```

```{code-cell} ipython3
F = Framework(Graph([[0,1]]), {0:[1,2], 1:[0,5]})
F.plot()
E = frameworks.Path(6)
E.plot()
D = Framework(Graph([[0,1]]), {0:[0,5], 1:[0,0]})
```

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
G.plot()
H = frameworks.Complete(7).graph()
H.plot()
print(H)
I = frameworks.Path(29).graph()
I.plot()
```

```{code-cell} ipython3
for i in range(5):
    G = frameworks.Complete(i+1).graph()
    G.plot()
```

```{code-cell} ipython3
import networkx as nx
a = nx.draw(H)
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3

```
