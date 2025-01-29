---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Using PyRigi for Rigidity Theory
Theory and Algorithms in Graph Rigidity and Algebraic Statistics

NII Shonan Meeting, Japan, September 3, 2024

```{code-cell} ipython3
# The import will work if:
#     - the tutorial is in the root folder of the package, or
#     - the package was installed using poetry,
#       see https://pyrigi.github.io/PyRigi/development/howto.html#dependencies, or
#     - the package is added to the sys.path using the following with the correct
#       path to the pyrigi root folder
import os, sys
sys.path.insert(0, os.path.abspath("../../.."))
from pyrigi import Graph
```

## Graph construction

+++

### List of edges

```{code-cell} ipython3
G1 = Graph([(0,1), (1,2), (2,3), (0,3)])
G1
```

### Lists of vertices and edges

```{code-cell} ipython3
G2 = Graph.from_vertices_and_edges([0, 2, 5, 'b', 7, 'a'], [(0,7), ('a',5)])
G2.add_edge('a',7)
```

```{code-cell} ipython3
G2.plot()
```

```{code-cell} ipython3
G2.vertex_list()
```

### Adjacency matrix

```{code-cell} ipython3
from sympy import Matrix
Graph.from_adjacency_matrix(Matrix([
    [0,1,1],
    [1,0,0],
    [1,0,0]])).plot()
```

### Graph drawer

```{code-cell} ipython3
from pyrigi.graph_drawer import GraphDrawer
Drawer = GraphDrawer()
```

```{code-cell} ipython3
Drawer.graph()
```

### Graph database

```{code-cell} ipython3
import pyrigi.graphDB as graphs
```

```{code-cell} ipython3
G3 = graphs.ThreePrism()
G3.plot()
G3.plot(layout="planar")
```

## Framework construction

```{code-cell} ipython3
from pyrigi import Framework
```

### Specifying realization

```{code-cell} ipython3
K4 = graphs.Complete(4)

F1 = Framework(K4, {0:[1,2], 1:[0,5], 2:[-1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F1
```

```{code-cell} ipython3
F1.plot()
```

Acessing the positions of a vertex

```{code-cell} ipython3
F1[2]
```

### Class methods

```{code-cell} ipython3
F2 = Framework.Simplicial(K4, 3)
```

```{code-cell} ipython3
F2
```

The dimension of a framework

```{code-cell} ipython3
F2.dim()
```

```{code-cell} ipython3
F2[2]
```

```{code-cell} ipython3
F2.translate([3,4,5])
```

```{code-cell} ipython3
F2[2]
```

### Framework database

```{code-cell} ipython3
import pyrigi.frameworkDB as frameworks
```

```{code-cell} ipython3
TP = frameworks.ThreePrism()
TP.plot()
```

```{code-cell} ipython3
TP_e = frameworks.ThreePrismPlusEdge()
TP_e.plot()
```

```{code-cell} ipython3
TP_flex = frameworks.ThreePrism('parallel')
TP_flex.plot()
```

## Rigidity properties

+++

### Infinitesimal rigidity

One of the main applications of PyRigi is to determine whether a framework is
infinitesimally rigid. 

```{code-cell} ipython3
print(TP.is_inf_rigid())
print(TP_e.is_inf_rigid())
print(TP_flex.is_inf_rigid())
```

```{code-cell} ipython3
print(TP.is_min_inf_rigid())
print(TP_e.is_min_inf_rigid())
```

```{code-cell} ipython3
TP_flex.rigidity_matrix()
```

```{code-cell} ipython3
TP_flex.nontrivial_inf_flexes()[0]
```


### Generic rigidity

We can also use PyRigi to investigate the infinitesimal and global rigidity of graphs.

```{code-cell} ipython3
G_TP = TP.graph()
G_TP.is_rigid()
```

```{code-cell} ipython3
G_TP.is_rigid(dim=1)
```

```{code-cell} ipython3
G_TP.is_rigid(dim=3, combinatorial=False)
```

```{code-cell} ipython3
G_TP.is_globally_rigid()
```

```{code-cell} ipython3
G_TP.is_globally_rigid(dim=1)
```

```{code-cell} ipython3
for H in G_TP.extension_sequence(return_solution=True):
    H.plot(canvas_height=2)
```

```{code-cell} ipython3
G4 = graphs.ThreePrismPlusEdge()
G4.plot()
```

```{code-cell} ipython3
G4.is_globally_rigid()
```

```{code-cell} ipython3
G4.is_redundantly_rigid()
```

```{code-cell} ipython3
for H in G_TP.all_k_extensions(0, only_non_isomorphic=True):
    H.plot()
    assert(H.is_rigid())
```

```{code-cell} ipython3
for H in graphs.ThreePrism().all_k_extensions(1, only_non_isomorphic=True):
    H.plot()
    assert(H.is_rigid())
```

### Equilibrium Stresses

PyRigi can also be used to compute equilibrium stresses. 

```{code-cell} ipython3
F = frameworks.Frustum(3)
inf_flex = F.inf_flexes()[0]
stress = F.stresses()[0]
F.plot(inf_flex=inf_flex, stress=stress)
```

The stress matrix criterion by Connelly (2005) states that a framework in $\RR^d$ with $n>d+2$ vertices is globally
rigid, if it possesses an equilibrium stress $\omega$ such that the associated stress matrix $\Omega(\omega)$ has rank $n-d-1$.

```{code-cell} ipython3
Omega = F.stress_matrix(stress)
Omega.rank()
```

The $3$-Frustum has $6>3+2$ vertices and its stress matrix has rank 3, so it is globally rigid in $\RR^d$.

## A globally rigid graph with 2 Penny realizations

In this section, we are going to construct a graph ``G`` which has two Penny realization. In other words,
there are two frameworks on ``G`` such that each edge has length 1 and each non-edge has length greater than 1. 

```{code-cell} ipython3
import sympy as sp
from fractions import Fraction
```

```{code-cell} ipython3
dx = sp.Matrix([1,0])
dy = sp.Matrix([0,sp.sqrt(3)/2])
half = Fraction(1,2)
```

```{code-cell} ipython3
p1 = {
    1 : 0*(dx)+0*(dy),
    2 : 1*(dx)+0*(dy),
    3 : 2*(dx)+0*(dy),
    4 : 3*(dx)+0*(dy),
    5 : 4*(dx)+0*(dy),
    6 : -1*half*(dx)+1*(dy),
    7 : 1*half*(dx)+1*(dy),
    8 : 3*half*(dx)+1*(dy),
    9 : 5*half*(dx)+1*(dy),
    10 : 7*half*(dx)+1*(dy),
    11 : 9*half*(dx)+1*(dy),
    12 : 4*(dx)+2*(dy),
    13 : 5*(dx)+2*(dy),
    14 : 9*half*(dx)+3*(dy),
    15 : 11*half*(dx)+3*(dy),
    16 : 5*(dx)+4*(dy),
    17 : 6*(dx)+4*(dy),
    18 : 9*half*(dx)+5*(dy),
    19 : 11*half*(dx)+5*(dy),
    20 : 4*(dx)+6*(dy),
    21 : 5*(dx)+6*(dy),
    22 : 7*half*(dx)+7*(dy),
    23 : 9*half*(dx)+7*(dy),
    24 : -1*(dx)+8*(dy),
    25 : 0*(dx)+8*(dy),
    26 : 1*(dx)+8*(dy),
    27 : 2*(dx)+8*(dy),
    28 : 3*(dx)+8*(dy),
    29 : 4*(dx)+8*(dy),
    30 : -1*half*(dx)+9*(dy),
    31 : 1*half*(dx)+9*(dy),
    32 : 3*half*(dx)+9*(dy),
    33 : 5*half*(dx)+9*(dy),
    34 : 7*half*(dx)+9*(dy),
    35 : 0*(dx)+2*(dy),
    36 : 2*(dx)+2*(dy),
    37 : -1*half*(dx)+7*(dy),
    38 : 3*half*(dx)+7*(dy),
    39 : -1*(dx)+4*(dy),
    40 : 0*(dx)+4*(dy),
    41 : 1*(dx)+4*(dy),
    42 : 2*(dx)+4*(dy),
    43 : -3*half*(dx)+5*(dy),
    44 : -1*half*(dx)+5*(dy),
    45 : 1*half*(dx)+5*(dy),
    46 : 3*half*(dx)+5*(dy),
    47 : -1*half*(dx)+3*(dy),
    48 : 3*half*(dx)+3*(dy),
    49 : -1*(dx)+6*(dy),
    50 : 1*(dx)+6*(dy),
}
p2 = {
    1 : 0*(dx)+0*(dy),
    2 : 1*(dx)+0*(dy),
    3 : 2*(dx)+0*(dy),
    4 : 3*(dx)+0*(dy),
    5 : 4*(dx)+0*(dy),
    6 : -1*half*(dx)+1*(dy),
    7 : 1*half*(dx)+1*(dy),
    8 : 3*half*(dx)+1*(dy),
    9 : 5*half*(dx)+1*(dy),
    10 : 7*half*(dx)+1*(dy),
    11 : 9*half*(dx)+1*(dy),
    12 : 4*(dx)+2*(dy),
    13 : 5*(dx)+2*(dy),
    14 : 9*half*(dx)+3*(dy),
    15 : 11*half*(dx)+3*(dy),
    16 : 5*(dx)+4*(dy),
    17 : 6*(dx)+4*(dy),
    18 : 9*half*(dx)+5*(dy),
    19 : 11*half*(dx)+5*(dy),
    20 : 4*(dx)+6*(dy),
    21 : 5*(dx)+6*(dy),
    22 : 7*half*(dx)+7*(dy),
    23 : 9*half*(dx)+7*(dy),
    24 : -1*(dx)+8*(dy),
    25 : 0*(dx)+8*(dy),
    26 : 1*(dx)+8*(dy),
    27 : 2*(dx)+8*(dy),
    28 : 3*(dx)+8*(dy),
    29 : 4*(dx)+8*(dy),
    30 : -1*half*(dx)+9*(dy),
    31 : 1*half*(dx)+9*(dy),
    32 : 3*half*(dx)+9*(dy),
    33 : 5*half*(dx)+9*(dy),
    34 : 7*half*(dx)+9*(dy),
    35 : 0*(dx)+2*(dy),
    36 : 2*(dx)+2*(dy),
    37 : -1*half*(dx)+7*(dy),
    38 : 3*half*(dx)+7*(dy),
    39 : 0*(dx)+4*(dy),
    40 : 1*(dx)+4*(dy),
    41 : 2*(dx)+4*(dy),
    42 : 3*(dx)+4*(dy),
    43 : -1*half*(dx)+5*(dy),
    44 : 1*half*(dx)+5*(dy),
    45 : 3*half*(dx)+5*(dy),
    46 : 5*half*(dx)+5*(dy),
    47 : 1*half*(dx)+3*(dy),
    48 : 5*half*(dx)+3*(dy),
    49 : 0*(dx)+6*(dy),
    50 : 2*(dx)+6*(dy),
}
```

```{code-cell} ipython3
E1 = [(1, 2), (1, 6), (1, 7), (2, 3), (2, 7), (2, 8), (3, 4), (3, 8), (3, 9), (4, 5), (4, 9), (4, 10), (5, 10), (5, 11), (6, 7), (6, 35), (7, 8), (7, 35), (8, 9), (8, 36), (9, 10), (9, 36), (10, 11), (10, 12), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14), (13, 15), (14, 15), (14, 16), (15, 16), (15, 17), (16, 17), (16, 18), (16, 19), (17, 19), (18, 19), (18, 20), (18, 21), (19, 21), (20, 21), (20, 22), (20, 23), (21, 23), (22, 23), (22, 28), (22, 29), (23, 29), (24, 25), (24, 30), (24, 37), (25, 26), (25, 30), (25, 31), (25, 37), (26, 27), (26, 31), (26, 32), (26, 38), (27, 28), (27, 32), (27, 33), (27, 38), (28, 29), (28, 33), (28, 34), (29, 34), (30, 31), (31, 32), (32, 33), (33, 34)]
E2 = [(35, 47), (36, 48), (37, 49), (38, 50)]
E3 = [(39, 40), (39, 43), (39, 44), (39, 47), (40, 41), (40, 44), (40, 45), (40, 47), (41, 42), (41, 45), (41, 46), (41, 48), (42, 46), (42, 48), (43, 44), (43, 49), (44, 45), (44, 49), (45, 46), (45, 50), (46, 50)]
```

```{code-cell} ipython3
G = Graph(E1+E2+E3)
```

```{code-cell} ipython3
F1 = Framework(G, p1)
F2 = Framework(G, p2)
```

```{code-cell} ipython3
F1.plot(vertex_labels=False, edge_coloring = {'black':E1, 'red':E2, 'blue':E3}, vertex_size=10, vertex_color='grey')
F2.plot(vertex_labels=False, edge_coloring = {'black':E1, 'red':E2, 'blue':E3}, vertex_size=10, vertex_color='grey')
```

Indeed, these realizations correspond to a Penny graph:

```{code-cell} ipython3
from itertools import combinations
for F in [F1, F2]:
    for u,v in combinations(G.vertex_list(), 2):
        d = (F[u]-F[v]).norm()
        if G.has_edge(u,v):
            assert(d == 1)
        else:
            assert(d > 1)
```

Still, the underlying graph is globally rigid. 

```{code-cell} ipython3
G.is_globally_rigid(dim=2)
```
