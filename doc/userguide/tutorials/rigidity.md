---
jupytext:
  formats: md:myst,ipynb
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

+++

This notebook illustrates how to use PyRigi for applications in rigidity theory using the
classes {class}`~.Graph` and {class}`~.Framework`.
It can be downloaded {download}`here <../../notebooks/rigidity.ipynb>`.

```{code-cell} ipython3
import pyrigi.frameworkDB as frameworks
import pyrigi.graphDB as graphs
from pyrigi import Graph, Framework
```


## Framework Construction

One way to construct a {prf:ref}`framework <def-framework>` is to provide a graph and a 
{prf:ref}`realization <def-realization>` to the constructor
{class}`~.Framework`. For instance, a framework ``F1`` on the complete graph on 4 vertices $K_4$
can be constructed in the following way.

```{code-cell} ipython3
K4 = graphs.Complete(4)

F1 = Framework(K4, {0:[1,2], 1:[0,5], 2:[-1,'1/2 * sqrt(5)'], 3:[1/2,'4/3']})
F1
```

The framework can then be visualized by calling the method {meth}`~.Framework.plot` on ``F1``,
see also tutorial [Plotting](plotting-tutorial).

```{code-cell} ipython3
F1.plot()
```

The position of a vertex in ``F1`` can be retrieved by calling

```{code-cell} ipython3
F1[2]
```

### Class methods

Alternatively, the realization can be specified by using a class method. For instance, {meth}`~.Framework.Simplicial` creates
a realization on the $d$-simplex. 

```{code-cell} ipython3
F2 = Framework.Simplicial(K4, 3)
F2
```

The dimension that a framework is embedded in can be accessed via the {meth}`~.Framework.dim` method.

```{code-cell} ipython3
F2.dim
```

```{code-cell} ipython3
F2[2]
```

The framework can be translated using the {meth}`~.Framework.translate` method.

```{code-cell} ipython3
F2.translate([3,4,5])
```

```{code-cell} ipython3
F2[2]
```

### Framework database

There exists a framework database from which certain frameworks can be imported.
A detailed tutorial of it can be accessed [here](tutorial-framework-database).

## Rigidity properties

+++

### Infinitesimal rigidity

One of the main applications of PyRigi is to determine whether a framework is
{prf:ref}`infinitesimally rigid <def-inf-rigid-framework>`. Take for example the following realization of the 3-prism.
We can determine whether it is infinitesimally rigid using the method {meth}`~.Framework.is_inf_rigid()`.

```{code-cell} ipython3
TP_rigid = frameworks.ThreePrism()
TP_rigid.plot()
TP_rigid.is_inf_rigid()
```

We can also create an {prf:ref}`infinitesimally flexible <def-inf-rigid-framework>`, but {prf:ref}`continuously rigid <def-cont-rigid-framework>` realization of the 3-prism using the parameter ``'parallel'``.

```{code-cell} ipython3
TP_parallel = frameworks.ThreePrism('parallel')
TP_parallel.plot()
TP_parallel.is_inf_rigid()
```

We check its rigidity using the method {meth}`~.Framework.is_inf_rigid`.
Finally, a {prf:ref}`continuously flexible <def-cont-rigid-framework>` realization can be created using the keyword ``'flexible'``.

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

In particular, the resulting framework is not {prf:ref}`minimally infinitesimally rigid <def-min-rigid-framework>`
anymore, even though the 3-prism was.

```{code-cell} ipython3
print(TP_rigid.is_min_inf_rigid())
print(TP_e.is_min_inf_rigid())
```

To investigate further properties of the framework, PyRigi can be used for computing the corresponding
{prf:ref}`rigidity matrix <def-rigidity-matrix>` using the method {meth}`~.Framework.rigidity_matrix`. 

```{code-cell} ipython3
TP_flexible.rigidity_matrix()
```

If you do not want to compute the infinitesimal flexes on your own, you can ask PyRigi to do it. The method
{meth}`~.Framework.nontrivial_inf_flexes` returns a list of {prf:ref}`nontrivial infinitesimal flexes <def-trivial-inf-flex>`
that all in the format of a matrix. 

```{code-cell} ipython3
inf_flexes = TP_flexible.nontrivial_inf_flexes()
inf_flexes[0]
```

We can verify that a vector is indeed an nontrivial infinitesimal flexes by calling the method
{meth}`~.Framework.is_nontrivial_flex`.

```{code-cell} ipython3
print(TP_flexible.is_nontrivial_flex(inf_flexes[0]))
print(TP_rigid.is_nontrivial_flex(inf_flexes[0]))
```

The list of {prf:ref}`trivial infinitesimal flexes <def-trivial-inf-flex>` can be accessed via the method
{meth}`~.Framework.trivial_inf_flexes`.

```{code-cell} ipython3
inf_flexes = TP_flexible.trivial_inf_flexes()
inf_flexes[0]
```


### Equilibrium Stresses

PyRigi can also be used to compute {prf:ref}`equilibrium stresses <def-equilibrium-stress>`.
This is done via the method {meth}`~.Framework.stresses`.

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
{meth}`~.Framework.is_stress`.

```{code-cell} ipython3
F.is_stress(stress)
```

The stress matrix associated to a framework for a given stress can be accessed via
the method {meth}`~.Framework.stress_matrix`.

```{code-cell} ipython3
Omega = F.stress_matrix(stress)
```

### Generic rigidity

We can also use PyRigi to investigate the {prf:ref}`generic rigidity<def-gen-rigid>` of graphs.
The underlying graph of a framework can be accessed
via the method {meth}`~.Framework.graph` and the rigidity of this graph can be determined via
{meth}`~.Graph.is_rigid`.

```{code-cell} ipython3
G_TP = TP_rigid.graph
G_TP.is_rigid()
```

The dimension in which its rigidity
is supposed to be investigated can be specified via the ``dim`` keyword.

```{code-cell} ipython3
G_TP.is_rigid(dim=1)
```

For dimensions greater than 2, no combinatorial rigidity criterion is known.
To still get a result, a randomized algorithm can be called using the command:

```{code-cell} ipython3
G_TP.is_rigid(dim=3, algorithm="randomized")
```

In fact, we can compute the maximal dimension, in which a graph is rigid using the method {meth}`~.Graph.max_rigid_dimension`.

```{code-cell} ipython3
G_TP.max_rigid_dimension()
```

Furthermore, we can compute the rigid components of a graph using {meth}`~.Graph.rigid_components`,
which is returned as a partition of vertices.

```{code-cell} ipython3
G = graphs.DoubleBanana()
G.rigid_components(dim=3, algorithm="randomized")
```

We can also investigate the (generic) ({prf:ref}`global <def-globally-rigid-graph>`)
rigidity of a graph using the method {meth}`~.Graph.is_globally_rigid`:

```{code-cell} ipython3
G_TP.is_globally_rigid()
```

and

```{code-cell} ipython3
G4 = graphs.ThreePrismPlusEdge()
G4.plot()
G4.is_globally_rigid()
```

To distinguish graphs from frameworks in the {meth}`~.Graph.plot` method, the vertices are colored differently by default.
The graph ``G4`` obtained by adding an edge to the 3-prism is globally rigid because it is 3-connected and
redundantly rigid. The redundant rigidity of a graph can be checked via {meth}`~.Graph.is_redundantly_rigid`.

```{code-cell} ipython3
G4.is_redundantly_rigid()
```

We can also investigate the global rigidity for other dimensions, too.

```{code-cell} ipython3
G_TP.is_globally_rigid(dim=1)
```

Finally, it may be useful to check whether a graph can be constructed via an {prf:ref}`extension sequence <def-k-extension>`,
which can be done using the method {meth}`~.Graph.extension_sequence`.

```{code-cell} ipython3
for H in G_TP.extension_sequence(return_type="graphs"):
    H.plot(canvas_height=2)
```

We can obtain all non-isomorphic $k$-{prf:ref}`extensions <def-k-extension>` of a graph using the method {meth}`~.Graph.all_k_extensions`.
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

### Sparsity

The {prf:ref}`(k,l)-sparsity <def-kl-sparse-tight>` of a graph can be checked using the method {meth}`~.Graph.is_kl_sparse`.

```{code-cell} ipython3
G = graphs.CompleteBipartite(3,3)
G.is_kl_sparse(2, 3)
```

Famously, the double banana is (3,6)-sparse, but not rigid. 

```{code-cell} ipython3
G = graphs.DoubleBanana()
print(G.is_kl_sparse(3, 6))
print(G.is_rigid(dim=3, algorithm="randomized"))
```

Similarly, it can be checked whether a graph is {prf:ref}`(k,l)-tight <def-kl-sparse-tight>`.

```{code-cell} ipython3
TP = graphs.ThreePrism()
TP.is_kl_tight(2, 3)
```

### Matroidal Properties

We can use PyRigi to check properties of graphs in the {prf:ref}`d-dimensional rigidity matroid <def-rigidity-matroid>` as well.
To check whether a graph is {prf:ref}`(in-)dependent <def-matroid>` in this matroid, we can call
{meth}`~.Graph.is_Rd_dependent` or {meth}`~.Graph.is_Rd_independent`.

```{code-cell} ipython3
TP.is_Rd_independent()
```

It can be checked whether a graph is a {prf:ref}`circuit <def-matroid>` in the d-dimensional rigidity matroid by calling {meth}`~.Graph.is_Rd_circuit`.

```{code-cell} ipython3
TP.is_Rd_circuit()
```

Finally, it can be determined whether the graph is $R_d$-{prf:ref}`closed <def-rank-function-closure>` by calling {meth}`~.Graph.is_Rd_closed`.

```{code-cell} ipython3
TP.is_Rd_closed(dim=1)
```
