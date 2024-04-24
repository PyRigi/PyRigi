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

**`PyRigi`**: {class}`~pyrigi.framework.Framework`,
{meth}`~.Framework.underlying_graph`,
{meth}`~.Framework.get_realization`
:::



