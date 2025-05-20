# NAC-Colorings

:::{prf:definition}
:label: def-nac

Let $G = (V,E)$ be a graph.
A _NAC-coloring_ is a coloring of the egdes in two colors, say red and blue, $\delta: E(G) \rightarrow \{red, blue\}$ such that
both colors occur (i.e. $\delta$ is surjective) and
every cycle of the graph is either monochromatic or it contains each color at least twice.

{{references}} {cite:p}`GraseggerLegerskySchicho1019`
:::


:::{prf:theorem}
:label: thm-nac

A connected graph has a {prf:ref}`flexible<def-cont-rigid-framework>` {prf:ref}`quasi-injective<def-realization>` realization in $\mathbb R^2$ if and only if it has a
NAC-coloring.

{{references}} {cite:p}`GraseggerLegerskySchicho1019{Thm 3.1}`
:::
