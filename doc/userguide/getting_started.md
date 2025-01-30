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

# Getting started

Jupyter notebook version of this page can be downloaded {download}`here <../notebooks/getting_started.ipynb>`.

## Installation

We have not reached a stable version yet,
but the latest version requiring at least Python 3.10
can be installed by
```
pip install pyrigi
```

Alternatively, one can clone/download the package
from [this GitHub repository](https://github.com/pyRigi/PyRigi).
Installation for development is done via [Poetry](dependencies). 
More detailed installation instructions depending on your operating system can be found [here](installation-guide).

+++

## Usage

Once the package is installed, the basic classes can be imported as follows:

```{code-cell} ipython3
from pyrigi import Graph, Framework
```

We can specify a graph by its list of edges, see [this tutorial](graph-tutorial) for more options.

```{code-cell} ipython3
G = Graph([(0,1), (1,2), (2,3), (0,3)])
G
```

```{code-cell} ipython3
G.plot()
```

Having graph `G`, we can construct a framework.

```{code-cell} ipython3
F = Framework(G, {0:[0,0], 1:[1,0], 2:[1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F
```

Notice that in order to keep the coordinates symbolic, they must be entered as strings (or SymPy expressions).

We can plot frameworks and graphs, see also the [Plotting](plotting-tutorial) tutorial. 

```{code-cell} ipython3
F.plot()
```

Positions of vertices can be read using square brackets:

```{code-cell} ipython3
F[2]
```

There are also some predefined graphs and frameworks, see {mod}`~pyrigi.graphDB` and {mod}`~pyrigi.frameworkDB` (also [this tutorial](tutorial-framework-database))

```{code-cell} ipython3
import pyrigi.frameworkDB as frameworks
```

```{code-cell} ipython3
TP_flex = frameworks.ThreePrism('parallel')
TP_flex.plot()
```

There is also a possibility to [draw a graph using mouse](graph-drawer-tutorial).  

## Rigidity properties

+++

Various rigidity properties can be checked by calling class methods, some examples are below.
See also [this tutorial](rigidity-tutorial). 

+++

### Infinitesimal rigidity

```{code-cell} ipython3
TP_flex.is_inf_rigid()
```

```{code-cell} ipython3
TP_flex.rigidity_matrix()
```

```{code-cell} ipython3
TP_flex.nontrivial_inf_flexes()
```

### Generic rigidity

```{code-cell} ipython3
G_TP = TP_flex.graph()
G_TP.is_rigid()
```

```{code-cell} ipython3
G_TP.is_rigid(dim=1)
```

```{code-cell} ipython3
G_TP.is_rigid(dim=3, algorithm="randomized")
```

```{code-cell} ipython3
G_TP.is_globally_rigid()
```

```{code-cell} ipython3
G_TP.is_globally_rigid(dim=1)
```

```{code-cell} ipython3
G_TP.is_redundantly_rigid()
```
