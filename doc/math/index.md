Basic definitions
-----------------

:::{prf:definition} Realization
:label: def-realization

Let $G=(V_G,E_G)$ be a simple graph, $\KK$ be a field and $d\in\NN$.
A $d$-dimensional *realization* of $G$ in $\KK^d$ is a map $p\colon V_G\rightarrow \KK^d$.

The realization $p$ is *quasi-injective* if $p(u)\neq p(v)$ for every edge $uv\in E_G$.
:::

:::{prf:definition} Framework
:label: def-framework

Let $G$ be a graph and let $p$ be a $d$-dimensional {prf:ref}`realization <def-realization>` of $G$.
The pair $(G, p)$ is a called a *framework*.

{{pyrigi}}:
{class}`~pyrigi.framework.Framework`,
{meth}`~.Framework.underlying_graph`,
{meth}`~.Framework.get_realization`
:::

:::{prf:definition} $(k, \ell)$-sparse and $(k, \ell)$-tight
:label: def-kl-sparsity

Let $G = (V_G, E_G)$ be a (multi)graph and let $k, \ell \in \NN$.
Set $n = |V_G|$ and $m = |E_G|$.
The graph $G$ is said to be *$(k, \ell)$-sparse* if every set of $n' \leq n$ vertices spans at most $kn' - \ell$ edges.
The graph $G$ is said to be *$(k, \ell)$-sparse* if it is $(k, \ell)$-sparse and $kn - \ell = m$.

_References:_ {cite:p}`Lee2008`
:::

:::{prf:theorem}
:label: thm-2-gen-rigidity

A graph $G = (V_G, E_G)$ is generically $2$-rigid if and only if ...

_References:_ {cite:p}`Geiringer1927`
{cite:p}`Laman1970`
:::

:::{bibliography}
:filter: docname in docnames
:::
