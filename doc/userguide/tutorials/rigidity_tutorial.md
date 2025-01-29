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

(rigidity-tutorial)=
# Using PyRigi for Rigidity Theory

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

## Framework construction

```{code-cell} ipython3
from pyrigi import Framework
```

### Specifying realization

One way to construct a framework is to provide a graph and a realization to the constructor
``Framework``. For instance, a framework ``F1`` on the complete graph on 4 vertices $K_4$
can be constructed in the following way.

```{code-cell} ipython3
from pyrigi import graphDB as graphs
K4 = graphs.Complete(4)

F1 = Framework(K4, {0:[1,2], 1:[0,5], 2:[-1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F1
```

The framework can then be visualized by calling the method ``plot`` on ``F1``.

```{code-cell} ipython3
F1.plot()
```

The position of a vertex in ``F1`` can be retrieved by calling

```{code-cell} ipython3
F1[2]
```

### Class methods

Alternatively, the realization can be specified by using a class method. For instance, ``Simplicial`` creates
a realization on the ``d``-simplex. 

```{code-cell} ipython3
F2 = Framework.Simplicial(K4, 3)
F2
```

The dimension of a framework can be accessed via the ``dim`` method.

```{code-cell} ipython3
F2.dim()
```

```{code-cell} ipython3
F2[2]
```

The framework can be translated using the ``translate`` method.

```{code-cell} ipython3
F2.translate([3,4,5])
```

```{code-cell} ipython3
F2[2]
```

### Framework database

Similar to the graph database, there exists a framework database. A detailed tutorial of it can be
accessed [here](tutorial-framework-database). The framework database can be imported via

the following command:
```{code-cell} ipython3
from pyrigi import frameworkDB as frameworks
```


## Rigidity properties

+++

### Infinitesimal rigidity

One of the main applications of PyRigi is to determine whether a framework is
infinitesimally rigid. Take for example a generic realization of the 3-prism. 

```{code-cell} ipython3
TP_rigid = frameworks.ThreePrism()
TP_rigid.plot()
TP_rigid.is_inf_rigid()
```

We can also create an infinitesimally flexible, but continuously rigid realization
of the 3-prism using the parameter ``'parallel'``.

```{code-cell} ipython3
TP_parallel = frameworks.ThreePrism('parallel')
TP_parallel.plot()
TP_parallel.is_inf_rigid()
```

Finally, a continuously flexible realization can be created using the keyword ``'flexible'``.

```{code-cell} ipython3
TP_flexible = frameworks.ThreePrism('flexible')
TP_flexible.plot()
TP_flexible.is_inf_rigid()
```

Adding an edge to the 3-prism changes its rigidity properties.

```{code-cell} ipython3
TP_e = frameworks.ThreePrismPlusEdge()
TP_e.plot()
TP_e.is_inf_rigid()
```

In particular, the resulting framework is not minimally infinitesimally rigid anymore, even though the 3-prism was.

```{code-cell} ipython3
print(TP_rigid.is_min_inf_rigid())
print(TP_e.is_min_inf_rigid())
```

To investigate further properties of the framework, PyRigi can be used for computing the corresponding rigidity matrix
using the method ``rigidity_matrix``. 

```{code-cell} ipython3
TP_flexible.rigidity_matrix()
```

If you do not want to compute the infinitesimal flexes on your own, you can ask PyRigi to do it. The method
``nontrivial_inf_flexes`` returns a list of infinitesimal flexes that all in the format of a matrix. 

```{code-cell} ipython3
inf_flexes = TP_flexible.nontrivial_inf_flexes()
inf_flexes[0]
```

We can verify that an infinitesimal flex is indeed a flex by calling

```{code-cell} ipython3
print(TP_flexible.is_nontrivial_flex(inf_flexes[0]))
print(TP_rigid.is_nontrivial_flex(inf_flexes[0]))
```

### Equilibrium Stresses

PyRigi can also be used to compute equilibrium stresses. 

```{code-cell} ipython3
F = frameworks.Frustum(3)
inf_flex = F.inf_flexes()[0]
stress = F.stresses()[0]
stress
```

We can visualize both the infinitesimal flexes and equilibrium stresses of a
framework by using the appropriate keywords.

```{code-cell} ipython3
F.plot(inf_flex=inf_flex, stress=stress)
```

Again, it can be verified that the stress indeed lies in the cokernel of the rigidity matrix by calling

```{code-cell} ipython3
F.is_stress(stress)
```

The stress matrix criterion by Connelly (2005) states that a framework in $\RR^d$ with $n>d+2$ vertices is globally
rigid, if it possesses an equilibrium stress $\omega$ such that the associated stress matrix $\Omega(\omega)$ has rank $n-d-1$.

```{code-cell} ipython3
Omega = F.stress_matrix(stress)
Omega.rank()
```

The $3$-Frustum has $6>3+2$ vertices and its stress matrix has rank 3, so it is globally rigid in $\RR^d$.

### Generic rigidity

We can also use PyRigi to investigate the (generic) infinitesimal and global rigidity of graphs.

```{code-cell} ipython3
G_TP = TP_rigid.graph()
G_TP.is_rigid()
```

Since the graph does come with a natural embedding, the dimension where its rigidity is supposed to be investigated
can be specified via the ``dim`` keyword.

```{code-cell} ipython3
G_TP.is_rigid(dim=1)
```

For dimensions greater than 2, there is not a combinatorial rigidity criterion yet. To still get a result, a framework with
randomized coordinates can be created using the command

```{code-cell} ipython3
G_TP.is_rigid(dim=3, combinatorial=False)
```

In fact, we can compute the maximal dimension, in which a graph is rigid.

```{code-cell} ipython3
G_TP.max_rigid_dimension()
```

Furthermore, we can compute the rigid components of a graph, which is returned as a partition of vertices.

```{code-cell} ipython3
G = graphs.DoubleBanana()
G.rigid_components(dim=3, combinatorial=False)
```

We can also investigate the (generic) global rigidity of a graph:

```{code-cell} ipython3
G_TP.is_globally_rigid()
```

and

```{code-cell} ipython3
G4 = graphs.ThreePrismPlusEdge()
G4.plot()
G4.is_globally_rigid()
```

To distinguish graphs from frameworks in the ``plot`` method, the vertices are colored differently by default.
The graph ``G4`` obtained by adding an edge to the 3-prism is globally rigid because it is 3-connected and
redundantly rigid:

```{code-cell} ipython3
G4.is_redundantly_rigid()
```

And can also investigate the global rigidity for other dimensions, too.

```{code-cell} ipython3
G_TP.is_globally_rigid(dim=1)
```

Finally, it may be useful to generate an extension sequence, which can be done for the 3-prism
using the method ``extension_sequence``.

```{code-cell} ipython3
for H in G_TP.extension_sequence(return_solution=True):
    H.plot(canvas_height=2)
```

We can obtain all non-isomorphic k-extensions of a graph using the method ``all_k_extensions``.
For the 3-prism we ensure that all of the 0-extensions are rigid:

```{code-cell} ipython3
for H in G_TP.all_k_extensions(0, only_non_isomorphic=True):
    H.plot()
    assert(H.is_rigid())
```

And we can do the same for the 1-extensions of the 3-prism.

```{code-cell} ipython3
for H in graphs.ThreePrism().all_k_extensions(1, only_non_isomorphic=True):
    H.plot()
    assert(H.is_rigid())
```

