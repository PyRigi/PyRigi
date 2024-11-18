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

# Graph Realizations (3D printing)

+++

This notebook can be downloaded {download}`here <../../notebooks/graph_realizations.ipynb>`.

```{code-cell} ipython3
# The import will work if the package was installed using pip.
from pyrigi import Graph, Framework
```

Method {meth}`.Framework.generate_stl_bars` allows one to generate simple bars of a framework that can be 3D printed.

We start with a definition of Graph and its Framework.


```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F.plot()
```

If the method {meth}`.Framework.generate_stl_bars` is called, it will generate bars 
in the working directory in STL format.

Sometimes we wish to alter the design of the bars...

```{code-cell} ipython3
F.generate_stl_bars(scale=20)
```


