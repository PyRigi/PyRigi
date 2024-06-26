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
    F.plot(edge_width=i+0.1, canvas_width=2*(i+1), canvas_height=i+1)
```

```{code-cell} ipython3
F1 = Framework(Graph([(0,1), (1,2), (2,3), (3,0)]), {0:(0,0), 1:(0,100), 2:(100,100), 3:(100,0)})
F2 = F1
F1.plot(canvas_width=5, canvas_height=2)
F2.plot()
```

```{code-cell} ipython3
F = Framework(Graph([(0,1), (1,2), (2,3), (3,0)]), {0:(0,0), 1:(0,30), 2:(1,30), 3:(1,0)})
F.plot()
F.plot(aspect_ratio=0.25)
```

```{code-cell} ipython3

```
