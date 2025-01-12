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

Let $G$ be a graph, let $d \in \NN$.
The graph $G$ is called _minimally (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`minimally (infinitesimally) d-rigid <def-min-rigid-framework>`.

{{pyrigi_crossref}} {meth}`~.Graph.is_min_rigid`
:::


:::{prf:theorem}
:label: thm-2-gen-rigidity

A graph $G = (V, E)$ with $|V|\geq 2$ is minimally (generically) $2$-rigid if and only if $G$ is {prf:ref}`(2,3)-tight <def-kl-sparse-tight>`.

{{references}} {cite:p}`Geiringer1927`
{cite:p}`Laman1970`
:::

:::{prf:theorem}
:label: thm-gen-rigidity-tight

Let $G = (V, E)$ be a minimally (generically) $d$-rigid graph with $|V|\geq d+1$. Then $G$ is $(d,\binom{d+1}{2})${prf:ref}`-tight <def-kl-sparse-tight>`.

{{references}} compare {cite:p}`Whiteley1996`
:::

:::{prf:theorem}
:label: thm-gen-rigidity-small-complete

Let $G = (V, E)$ be a graph with $|V|\leq d+1$. Then $G$ is minimally (generically) $d$-rigid if and only if $G$ is a complete graph.

{{references}} compare {cite:p}`GraverServatiusServatius1993{Lem 2.6.1}`
:::

:::{prf:theorem}
:label: thm-probabilistic-rigidity-check

Let $G = (V, E)$ be a graph and let $F=(G,p)$ be a framework with a random parametrization $p$ with coordinates between 1 and some $N$.
If $F$ is (minimally) infinitesimally $d$-rigid, then $G$ is (minimally) $d$-rigid.
If $F$ is not (minimally) infinitesimally $d$-rigid, then $G$ is not (minimally) $d$-rigid with probability $1-(dn-\binom{d+1}{2})/N$.
In other words the probability of a false negative is $(dn-\binom{d+1}{2})/N$.
{{references}} {cite:p}`Gortler2010`
:::

:::{prf:definition} Rigid components in $\RR^d$
:label: def-rigid-components

Let $G$ be a graph, let $d \in \NN$.
The _d-rigid components_ of $G$ are the maximal vertex-induced subgraphs 
of $G$ that are {prf:ref}`d-rigid <def-gen-rigid>`.

{{pyrigi_crossref}} {meth}`~.Graph.rigid_components`
:::
