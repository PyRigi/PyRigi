# Periodic Frameworks

:::{prf:definition} Periodic framework
:label: def-periodic-framework
We consider a {prf:ref}`$\mathbb{Z}^d$-gain graph <def-gain-graph>` $G=(V,E)$ and a full-rank lattice given by basis vectors $\Lambda_1,\dots, \Lambda_d \in \mathbb{R}^d$. They define the _fundamental parallelipiped_ or _unit cell_ $\mathbb{R}^d/\Lambda$ for the lattice $\mathbb{Lambda}=(\Lambda_1 \dots \Lambda_d)$. Also consider the realization $p:V\rightarrow \mathbb{R}^d$. We often view the vertex orbits in the unit cell by restricting $p$ to $\mathbb{R}^d/\Lambda$. A _$d$-periodic bar-joint framework_ is then given by the tuple $(G, p, \Lambda)$.

The _edge vector_ associated with a gain edge $(u, v, \gamma)$ is given by
\[p(v)+\Lambda\cdot \gamma-p(u).\]

{{pyrigi_crossref}} {class}`~.PeriodicFramework`
:::
