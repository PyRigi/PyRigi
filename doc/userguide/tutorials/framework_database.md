---
jupytext:
  formats: ipynb,md:myst
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

(tutorial-framework-database)=
# Database of frameworks

+++

This notebook can be downloaded {download}`here <../../notebooks/framework_database.ipynb>`.

+++

There are several predefined frameworks in {mod}`pyrigi.frameworkDB`.

```{code-cell} ipython3
# The import will work if the package was installed using pip.
import pyrigi.frameworkDB as frameworks
```

## Complete frameworks

+++

{func}`~.frameworkDB.Complete` returns $d$-dimensional complete frameworks.

```{code-cell} ipython3
frameworks.Complete(2)
```

```{code-cell} ipython3
frameworks.Complete(3, d=1)
```

```{code-cell} ipython3
frameworks.Complete(4, d=3)
```

```{code-cell} ipython3
K4 = frameworks.Complete(4, d=2)
print(K4)
K4.plot()
```

Currently, for $d\geq 3$, the number of vertices must be at most $d+1$ so the graph can be realized as a simplex.

```{code-cell} ipython3
try:
    frameworks.Complete(5, d=3)
except ValueError as e:
    print(e)
```

## Complete bipartite frameworks

+++

{func}`~.frameworkDB.CompleteBipartite` returns 2-dimensional complete bipartite frameworks.

```{code-cell} ipython3
K34 = frameworks.CompleteBipartite(3,3)
K34.plot()
K34.is_inf_rigid()
```

The first construction of a flexible realization by Dixon places one part on the $x$-axis and the other part on the $y$-axis.

```{code-cell} ipython3
K34_dixonI = frameworks.CompleteBipartite(3,3,'dixonI')
K34_dixonI.plot()
K34_dixonI.is_inf_flexible()
```

## Cycle frameworks

+++

{func}`~.frameworkDB.Cycle` returns $d$-dimensional frameworks on cycle graphs.
The restriction on the number of vertices w.r.t. the dimension is the same as for complete frameworks.

```{code-cell} ipython3
C5 = frameworks.Cycle(5)
print(C5)
C5.plot()
```

```{code-cell} ipython3
frameworks.Cycle(5,d=1)
```

```{code-cell} ipython3
frameworks.Cycle(5,d=4)
```

## Path frameworks

+++

{func}`~.frameworkDB.Path` returns $d$-dimensional frameworks on path graphs.
The restriction on the number of vertices w.r.t. the dimension is the same as for complete frameworks.

```{code-cell} ipython3
P5 = frameworks.Path(5)
print(P5)
P5.plot()
```

```{code-cell} ipython3
frameworks.Path(5,d=1)
```

```{code-cell} ipython3
frameworks.Path(5,d=4)
```

## 3-prism

+++

A general realization of 3-prism.

```{code-cell} ipython3
TP = frameworks.ThreePrism()
TP.plot()
TP.is_inf_rigid()
```

Infinitesimally flexible, but continuously rigid realization.

```{code-cell} ipython3
TP = frameworks.ThreePrism('parallel')
TP.plot()
TP.is_inf_rigid()
```

Continuously flexible realization.

```{code-cell} ipython3
TP = frameworks.ThreePrism('flexible')
TP.plot()
TP.is_inf_rigid()
```

## Further frameworks

```{code-cell} ipython3
Diamond = frameworks.Diamond()
print(Diamond)
Diamond.plot()
```

```{code-cell} ipython3
Square = frameworks.Square()
print(Square)
Square.plot()
```

```{code-cell} ipython3
frameworks.K33plusEdge().plot()
```

```{code-cell} ipython3
frameworks.ThreePrismPlusEdge().plot()
```
