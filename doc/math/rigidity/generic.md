# Generic Rigidity

:::{prf:definition} Generic realization and framework
:label: def-gen-realization

Let $G$ be a graph. A $d$-dimensional {prf:ref}`realization <def-realization>` $p$ of $G$ whose coordinates are algebraically independent is called _generic_.
A {prf:ref}`framework <def-framework>` $(G, p)$, where $p$ is generic, is called a _generic framework_.
:::

:::{prf:definition} Generically rigid graph
:label: def-gen-rigid

Let $G$ be a graph and $d \in \NN$.
The graph $G$ is called _(generically) $d$-rigid_ if any {prf:ref}`generic d-dimensional framework <def-gen-realization>` $(G, p)$ is {prf:ref}`rigid <def-cont-rigid-framework>`; this is equivalent to $(G, p)$ being {prf:ref}`infinitesimally rigid <def-inf-rigid-framework>`.

{{pyrigi_crossref}} {meth}`~.Graph.is_rigid`
:::



:::{prf:definition} Minimally generically rigid graphs
:label: def-min-rigid-graph

Let $G$ be a graph, let $d, k \in \NN$.
The graph $G$ is called _minimally (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`minimally (infinitesimally) d-rigid <def-min-rigid-framework>`.

{{pyrigi_crossref}} {meth}`~.Graph.is_min_rigid`
:::


:::{prf:theorem}
:label: thm-2-gen-rigidity

A graph $G = (V, E)$ is minimally (generically) $2$-rigid if and only if $G$ is {prf:ref}`(2,3)-tight <def-kl-sparse-tight>`.

{{references}} {cite:p}`Geiringer1927`
{cite:p}`Laman1970`
:::


