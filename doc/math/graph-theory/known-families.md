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

Here we introduce several known graphs that can be accessed in the PyRigi graph database.

:::{prf:definition} $n$-Frustum
:label: def-n-frustum

Assume that $n\geq 3$. The graph $G=(V,E)$ is called the _$n$-Frustum_ if it has the following properties: 
* $V = V_1\sqcup V_2$ with $|V_1|=|V_2|=n$.
* $G[V_1]$ and $G[V_2]$ are cycle graphs.
* For each vertex $v_1$ in $V_1$ there is exactly one edge $(v_1,v_2)$ in $E$ for some $v_2 \in V_2$.

{{pyrigi_crossref}} {meth}`~.graphDB.Frustum`
{meth}`~.frameworkDB.Frustum`
:::