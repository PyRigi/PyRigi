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
```

```{code-cell} ipython3
from pyrigi import Graph

G = Graph([(0,1), (1,2), (2,3), (0,3)])
H = Graph([(0,1)])

print(G)
G.plot()
H.plot()
```

```{code-cell} ipython3

```
