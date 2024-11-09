# Global Rigidity

:::{prf:definition}
:label: def-globally-rigid-graph

A graph $G = (V,E)$ is called _globally $d$-rigid_, if for every {prf:ref}`generic <def-gen-realization>` {prf:ref}`realization <def-realization>` $p:V \rightarrow \mathbb{R}^d$ all {prf:ref}`equivalent <def-equivalent-framework>` realizations $p':V\rightarrow \mathbb{R}$ are {prf:ref}`congruent <def-equivalent-framework>` to $p$.

{{references}} {cite:p}`Jackson2005`
:::


:::{prf:theorem}
:label: thm-globally-redundant-3connected

A graph $G$ is globally $2$-rigid if and only if it either is a
complete graph on at most three vertices or it is $3$-connected and {prf:ref}`redundantly rigid<def-redundantly-rigid-graph>`.

{{references}} {cite:p}`Jackson2005{Thm 7.1}`
:::

:::{prf:theorem}
:label: thm-globally-mindeg6-dim2

Let $G$ be a $6$-connected {prf:ref}`2-rigid <def-gen-rigid>` graph. Then $G$ is globally $2$-rigid.

{{references}} {cite:p}`Jackson2005{Thm 7.2}`
:::
