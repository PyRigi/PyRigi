# Gain graphs

:::{prf:definition} Directed Multigraph
:label: def-directed-multigraph

A _directed multidigraph_ is a directed graph $G=(V,E)$ with a multiset of ordered pairs $E\subset V\times V$
called _directed edges_, which is permitted to have _loops_, i.e., edges starting and ending in the same vertex,
and multiple _parallel edges_, i.e. directed edges with the same source and target nodes.

:::

:::{prf:definition} Gain graph
:label: def-gain-graph

A _$\Gamma$-gain graph_ is a pair $(G,\psi)$ where $G=(V,E)$ is a {prf:ref}`directed multigraph <def-directed-multigraph>`
and $\psi:E \rightarrow \Gamma$ is an edge labeling such that
1. For all {prf:ref}`loops <def-directed-multigraph>` $e=(u,u)\in E$ $\psi(e)\neq\text{id}_\Gamma$
2. For all {prf:ref}`parallel edges <def-directed-multigraph>` $e,f\in E$ with end vertices $u,v\in V$ $\psi(e)\neq \psi(f)$ if $e$ and $f$ have the same direction, 
and $\psi(e)\neq \psi(f)^{-1}$ if $e$ and $f$ have opposite direction.

A $\Gamma$-gain graph is called balanced if for all {prf:ref}`closed walks <def-directed-multigraph>`

{{pyrigi_crossref}} {class}`~pyrigi.GainGraph`
:::