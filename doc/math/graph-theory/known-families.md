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

(known-families)=
# Known Families of Graphs

Here we introduce several known graphs that can be accessed in the PyRigi graph database. To access them, we need to import the `graphDB`:

```{code-cell} ipython3
from pyrigi import graphDB
```

:::{prf:definition} $n$-Frustum
:label: def-n-frustum

The graph $G=(V,E)$ is called the _$n$-Frustum_ for $n\geq 3$ if $V = V_1\sqcup V_2$ such that $|V_1|=|V_2|=n$ with the following properties: 
* $G[V_1]$ and $G[V_2]$ are cycle graphs.
* For each vertex $v_1$ in $V_1$ there is exactly one edge $(v_1,v_2) in $E$ for $v_2\in V_2$.

Typically, the $n$-Frustum is realized as a planar framework two regular $n$-gons that are contained in each other. An example of a 3-Frustum is depicted below.


{{pyrigi_crossref}} {meth}`~.graphDB.Frustum`
{meth}`~.frameworkDB.Frustum`
:::

```{code-cell} ipython3
G = graphDB.Frustum(3)
G.plot()
```