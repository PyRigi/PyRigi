# Known Families of Graphs

Here we introduce several known graphs that can be accessed in the PyRigi graph database.

:::{prf:definition} $n$-Frustum
:label: def-n-frustum

Assume that $n\geq 3$. The graph $G=(V,E)$ is called the _$n$-Frustum_ if it is the Cartesian product $G=C_n\,\square \, K_2$ of a cycle graph $C_n$ on $n$ vertices and the complete graph $K_2$ on two vertices.

{{pyrigi_crossref}} {func}`.graphDB.Frustum`
{func}`.frameworkDB.Frustum`
:::
