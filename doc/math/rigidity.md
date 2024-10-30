(definitions)=
# Rigidity Theory


Here we list the very basic definitions used in Rigidity Theory.
For further topics, see the links below.


:::{prf:definition} Realization
:label: def-realization

Let $G=(V,E)$ be a simple graph (i.e. no multi edges and no loops) and $d\in\NN$.
A $d$-dimensional _realization_ of $G$ is a map $p\colon V\rightarrow \RR^d$.
For convenience, for $v \in V$ we may denote $p(v)$ by $p_v$.

The realization $p$ is _quasi-injective_ if $p(u)\neq p(v)$ for every edge $uv\in E$.

{{pyrigi_crossref}} {meth}`~.Framework.is_injective`
{meth}`~.Framework.is_quasi_injective`
:::


:::{prf:definition} Framework
:label: def-framework

Let $G$ be a graph and let $p$ be a $d$-dimensional {prf:ref}`realization <def-realization>` of $G$.
The pair $(G, p)$ is a called a _$d$-dimensional framework_.

{{pyrigi_crossref}} {class}`~pyrigi.framework.Framework`
{meth}`~.Framework.graph`
{meth}`~.Framework.realization`
:::


:::{prf:definition} Equivalent and congruent frameworks
:label: def-equivalent-framework

Two $d$-dimensional {prf:ref}`frameworks <def-framework>` $(G, p)$ and $(G, p')$ with $G = (V, E)$ are called _equivalent_ if

\begin{equation*}
 \left\| p_u - p_v \right\| = \left\| p'_u - p'_v \right\|
 \quad \text{ for all } uv \in E \,.
\end{equation*}

Two $d$-dimensional {prf:ref}`frameworks <def-framework>` $(G, p)$ and $(G, p')$ with $G = (V, E)$ are called _congruent_ if

\begin{equation*}
 \left\| p_u - p_v \right\| = \left\| p'_u - p'_v \right\|
 \quad \text{ for all } u, v \in V \,.
\end{equation*}
:::

:::{prf:definition} Continuous flexes
:label: def-flex

Let $(G, p)$ be a $d$-dimensional {prf:ref}`framework <def-framework>` with $G = (V, E)$.
A _continuous flex_ is a continuous map $\alpha \colon [0, 1] \rightarrow (\RR^{d})^V$ such that

* $\alpha(0) = p$;
* $(G, p)$ and $(G, \alpha(t))$ are equivalent for every $t \in [0,1]$.

A continuous flex is called _trivial_ if $(G, p)$ and $(G, \alpha(t))$ are congruent for every $t \in [0,1]$.
:::


:::{prf:definition} Continuously rigid frameworks
:label: def-cont-rigid-framework

A {prf:ref}`framework <def-framework>` $(G, p)$ is called _continuously rigid_ (from now on, simply _rigid_) if each of its continuous flexes is trivial.
A {prf:ref}`framework <def-framework>` $(G, p)$ is called _flexible_ if it is not rigid.
:::



## Further topics
:::{toctree}
:maxdepth: 2
rigidity/infinitesimal
rigidity/generic
rigidity/global
rigidity/redundant
:::