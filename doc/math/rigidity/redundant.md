# Redundant Rigidity

:::{prf:definition} Redundantly rigid frameworks
:label: def-redundantly-rigid-framework

Let $(G,p)$ be a $d$-dimensional {prf:ref}`framework <def-framework>` and let $k \in \NN$.
The framework $(G, p)$ is called

* _redundantly (infinitesimally) $d$-rigid_ if removing any edge from $G$ yields an ({prf:ref}`infinitesimally <def-inf-rigid-framework>`) {prf:ref}`rigid framework <def-cont-rigid-framework>`;
* _vertex redundantly (infinitesimally) $d$-rigid_ if removing any vertex from $G$ yields an ({prf:ref}`infinitesimally <def-inf-rigid-framework>`) {prf:ref}`rigid framework <def-cont-rigid-framework>`;
* _$k$-redundantly (infinitesimally) $d$-rigid_ if removing any set of at most $k$ edges from $G$ yields an ({prf:ref}`infinitesimally <def-inf-rigid-framework>`) {prf:ref}`rigid framework <def-cont-rigid-framework>`;
* _$k$-vertex redundantly (infinitesimally) $d$-rigid_ if removing any set of at most $k$ vertices from $G$ yields an ({prf:ref}`infinitesimally <def-inf-rigid-framework>`) {prf:ref}`rigid framework <def-cont-rigid-framework>`.

{{pyrigi_crossref}} {meth}`~.Framework.is_redundantly_inf_rigid`
:::

:::{prf:definition} Redundantly generically rigid graphs
:label: def-redundantly-rigid-graph

Let $G$ be a graph, let $d, k \in \NN$.
The graph $G$ is called

* _redundantly (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`redundantly (infinitesimally) d-rigid <def-redundantly-rigid-framework>`;
* _vertex redundantly (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`vertex redundantly (infinitesimally) d-rigid <def-redundantly-rigid-framework>`;
* _$k$-redundantly (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`k-redundantly (infinitesimally) d-rigid <def-redundantly-rigid-framework>`;
* _$k$-vertex redundantly (generically) $d$-rigid_ if a (equivalently, any) {prf:ref}`generic framework <def-gen-realization>` $(G, p)$ is {prf:ref}`k-vertex redundantly (infinitesimally) d-rigid <def-redundantly-rigid-framework>`.

Note, that the word generically is often omitted when talking about graphs.

{{pyrigi_crossref}} {meth}`~.Graph.is_redundantly_rigid`
{meth}`~.Graph.is_vertex_redundantly_rigid`
{meth}`~.Graph.is_k_redundantly_rigid`
{meth}`~.Graph.is_k_vertex_redundantly_rigid`
:::


:::{prf:definition} Minimally redundantly generically rigid graphs
:label: def-min-redundantly-rigid-graph

Let $G$ be a graph, let $d, k \in \NN$.
The graph $G$ is called

* _minimally redundantly (generically) $d$-rigid_ if it is {prf:ref}`redundantly (generically) d-rigid<def-redundantly-rigid-graph>` and there is an edge such that the graph obtained by deleting this edge is not redundantly (generically) $d$-rigid any more.
* _minimally vertex redundantly (generically) $d$-rigid_ if it is {prf:ref}`vertex redundantly (generically) d-rigid<def-redundantly-rigid-graph>` and there is an edge such that the graph obtained by deleting this edge is not vertex redundantly (generically) $d$-rigid any more.
* _minimally $k$-redundantly (generically) $d$-rigid_ if it is {prf:ref}`k-redundantly (generically) d-rigid<def-redundantly-rigid-graph>` and there is an edge such that the graph obtained by deleting this edge is not $k$-redundantly (generically) $d$-rigid any more.
* _minimally $k$-vertex redundantly (generically) $d$-rigid_ if it is {prf:ref}`k-vertex redundantly (generically) d-rigid<def-redundantly-rigid-graph>` and there is an edge such that the graph obtained by deleting this edge is not $k$-vertex redundantly (generically) $d$-rigid any more.

Note, that the word generically is often omitted when talking about graphs.

{{pyrigi_crossref}} {meth}`~.Graph.is_min_redundantly_rigid`
{meth}`~.Graph.is_min_vertex_redundantly_rigid`
{meth}`~.Graph.is_min_k_redundantly_rigid`
{meth}`~.Graph.is_min_k_vertex_redundantly_rigid`
:::

## (Edge) Redundancy
:::{prf:theorem}
:label: thm-red-min-deg

Let $G = (V, E)$ be a {prf:ref}`k-redundantly d-rigid <def-redundantly-rigid-graph>` graph with at least $d+k+1$ vertices,
then the minimum degree of $G$ is at least $d+k$.

{{references}} {cite:p}`Jordan2021{Lem 2}`
:::


:::{prf:theorem}
:label: thm-redundant-edge-subset

A graph $G = (V, E)$ is {prf:ref}`k-redundantly d-rigid <def-redundantly-rigid-graph>` if and only if the removal of any set of $k$ edges yields a $d$-rigid graph.

{{references}} {cite:p}`YuAnderson2009{Obs 1}`
:::


:::{prf:theorem}
:label: thm-k-edge-redundant-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`k-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 6 (k + 1) + 23$ and let $k \geq 3$. Then
\begin{equation*}
  |E| \geq \left\lceil \frac{k + 2}{2} |V| \right\rceil \,.
\end{equation*}

{{references}} {cite:p}`Jordan2021{Thm 6}`
:::


:::{prf:theorem}
:label: thm-2-edge-redundant-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`2-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 5$. Then
\begin{equation*}
  |E| \geq 2 |V| \,.
\end{equation*}

{{references}} compare  {cite:p}`Jordan2021{Thm 7}`
:::


:::{prf:theorem}
:label: thm-2-edge-redundant-edge-bound-dim3

Let $G = (V, E)$ be a {prf:ref}`2-redundantly 3-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 14$. Then
\begin{equation*}
  |E| \geq 3 |V| - 4 \,.
\end{equation*}

{{references}} {cite:p}`JordanPostonRoach2022{Thm 4.5}`
:::


:::{prf:theorem}
:label: thm-k-edge-redundant-edge-bound-dim3

Let $G = (V, E)$ be a {prf:ref}`k-redundantly 3-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 12 (k + 1) + 10$ where $|V|$ is even and $k \geq 4$. Then
\begin{equation*}
  |E| \geq \left\lceil \frac{k + 3}{2} |V| \right\rceil \,.
\end{equation*}

{{references}} {cite:p}`JordanPostonRoach2022{Thm 4.9}`
:::


:::{prf:theorem}
:label: thm-minimal-1-edge-redundant-upper-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`minimally 1-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V| \geq 7$. Then
\begin{equation*}
  |E| \leq 3 |V| - 9 \,.
\end{equation*}

{{references}} {cite:p}`Jordan2016{Sec 3.6.1}`
:::


## Vertex Redundancy
:::{prf:theorem}
:label: thm-vertex-red-min-deg

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` graph with at least $d+k+1$ vertices,
then the minimum degree of $G$ is at least $d+k$.

{{references}} {cite:p}`Jordan2021{Lem 2}`
:::

:::{prf:theorem}
:label: thm-redundant-vertex-subset

Let $G = (V, E)$ be a graph with at least $k+2$ vertices.
Then $G$ is {prf:ref}`k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` if and only if the removal of any set of $k$ vertices yields a $d$-rigid graph.

{{references}} {cite:p}`KaszanitzkyKiraly2015{Lem 1}`
{cite:p}`YuAnderson2009{Obs 6}`
:::

:::{prf:theorem}
:label: thm-k-vertex-redundant-edge-bound-general

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq d^2+d+k+1$. Then
\begin{equation*}
  |E| \geq d |V| - \binom{d + 1}{2} + k d + max \left\{ 0, \left\lceil k - \frac{d + 1}{2} \right\rceil \right\} \,.
\end{equation*}

{{references}} {cite:p}`KaszanitzkyKiraly2015{Thm 5}`
:::


:::{prf:theorem}
:label: thm-k-vertex-redundant-edge-bound-general2

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq d + k + 1$ and let $k \geq d + 1$. Then
\begin{equation*}
  |E| \geq \left\lceil \frac{d + k}{2} |V| \right\rceil \,.
\end{equation*}

{{references}} {cite:p}`KaszanitzkyKiraly2015{Thm 6}`
:::


:::{prf:theorem}
:label: thm-1-vertex-redundant-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`1-vertex-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 5$. Then
\begin{equation*}
  |E| \geq 2 |V| - 1 \,.
\end{equation*}

{{references}} {cite:p}`Servatius1989`
{cite:p}`SummersYuAnderson2008{Lem 1}`
:::


:::{prf:theorem}
:label: thm-2-vertex-redundant-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`2-vertex-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 6$. Then
\begin{equation*}
  |E| \geq 2 |V| + 2 \,.
\end{equation*}

{{references}} {cite:p}`MotevallianYuAnderson2014{Lem 4.9}`
:::


:::{prf:theorem}
:label: thm-k-vertex-redundant-edge-bound-dim2

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly 2-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 6 (k + 1) + 23$ and let $k \geq 3$. Then
\begin{equation*}
  |E| \geq \left\lceil \frac{k + 2}{2} |V| \right\rceil \,.
\end{equation*}

{{references}} {cite:p}`Jordan2021{Thm 5}`
:::


:::{prf:theorem}
:label: thm-3-vertex-redundant-edge-bound-dim3

Let $G = (V, E)$ be a {prf:ref}`3-vertex-redundantly 3-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 15$. Then
\begin{equation*}
  |E| \geq 3 |V| + 5 \,.
\end{equation*}

{{references}} {cite:p}`JordanPostonRoach2022{Lem 2.12}`
:::


:::{prf:theorem}
:label: thm-k-vertex-redundant-edge-bound-dim3

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly 3-rigid <def-redundantly-rigid-graph>` graph with $|V|\geq 12 (k + 1) + 10$ where $|V|$ is even and $k \geq 4$. Then
\begin{equation*}
  |E| \geq \left\lceil \frac{k + 3}{2} |V| \right\rceil \,.
\end{equation*}

{{references}} {cite:p}`JordanPostonRoach2022{Thm 3.3}`
:::


:::{prf:theorem}
:label: thm-minimal-k-vertex-redundant-upper-edge-bound

Let $G = (V, E)$ be a {prf:ref}`minimally k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` graph. Then
\begin{equation*}
  |E| \leq (d + k) |V| - \binom{d + k + 1}{2} \,.
\end{equation*}

{{references}} {cite:p}`KaszanitzkyKiraly2015{Thm 7}`
:::


:::{prf:theorem}
:label: thm-minimal-k-vertex-redundant-upper-edge-bound-dim1

Let $G = (V, E)$ be a {prf:ref}`minimally k-vertex-redundantly 1-rigid <def-redundantly-rigid-graph>` graph with $|V| \geq 3 (k + 1) - 1$. Then
\begin{equation*}
  |E| \leq (k + 1) |V| - (k + 1)^2 \,.
\end{equation*}

{{references}} {cite:p}`KaszanitzkyKiraly2015{Thm 8}`
:::


:::{prf:theorem}
:label: thm-vertex-implies_edge

Let $G = (V, E)$ be a {prf:ref}`k-vertex-redundantly d-rigid <def-redundantly-rigid-graph>` graph with at least $d+k+1$ vertices,
then $G$ is {prf:ref}`k-redundantly d-rigid <def-redundantly-rigid-graph>`.

{{references}} {cite:p}`Jordan2021{Thm 1}`
:::
