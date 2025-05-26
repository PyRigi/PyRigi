# NAC-Colorings

NAC-colorings are a concept introduced in {cite:p}`GraseggerLegerskySchicho2019`
and shown to be interesting for deciding the existence and for constructing flexible realizations of graphs.

:::{prf:definition} NAC-coloring
:label: def-nac

Let $G = (V,E)$ be a graph.
A _NAC-coloring_ is a coloring of the edges in two colors, say red and blue, $\delta \colon E(G) \rightarrow \{\tred, \tblue\}$ such that
both colors occur (i.e. $\delta$ is surjective) and
every cycle of the graph is either monochromatic or it contains each color at least twice.

{{references}} {cite:p}`GraseggerLegerskySchicho2019`
:::


:::{prf:lemma}
:label: lem-color-components

Let $G=(V,E)$ be a graph and let $\delta\colon E \rightarrow \{\tred, \tblue\}$ be a
surjective edge coloring.
Let $E_r$ and $E_b$ be the red and blue edges of $E$, respectively.
Then $\delta$ is a NAC-coloring if and only if the connected components of the edge-induced subgraphs $G[E_r]$ and $G[E_b]$
are indeed vertex-induced subgraphs of G.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Thm 2.4}`
:::


:::{prf:theorem}
:label: thm-nac

A connected graph has a {prf:ref}`flexible<def-cont-rigid-framework>` {prf:ref}`quasi-injective<def-realization>` realization in $\mathbb R^2$ if and only if it has a
NAC-coloring.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Thm 3.1}`
:::


:::{prf:theorem}
:label: thm-stable-separating-set

Let $G$ be a connected graph. If $G$ has a {prf:ref}`stable<def-stable-set>` {prf:ref}`separating set<def-separating-set>`, than $G$ has a NAC-coloring.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Thm 4.4}`
:::


:::{prf:corollary}
:label: cor-v-not-in-triangle

Let $G$ be a connected graph with at least 2 edges. If $G$ has a vertex that is not contained in any triangle subgraph of $G$, than $G$ has a NAC-coloring.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Cor 4.5}`
:::


:::{prf:theorem}
:label: thm-2-tree

A {prf:ref}`minimally 2-rigid<def-min-rigid-graph>` graph with more than one vertex has a NAC-coloring if and only if it is not a {prf:ref}`2-tree<def-2-tree>`.

{{references}} {cite:p}`ClinchGaramvölgyiEtAl2024{Thm 1.2}`
:::


:::{prf:theorem}
:label: thm-flexible-edge-bound

Let $G$ be a connected graph with $n$ vertices. If $G$ has a {prf:ref}`flexible<def-cont-rigid-framework>` realization, then $|E(G)|\leq \frac{n(n-1)}{2}-(n-2)$.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Thm 4.7}`
:::


:::{prf:definition} Cartesian NAC-coloring
:label: def-cartesian-nac

A NAC-coloring of $G$ is called _Cartesian_, if no two distinct vertices of $G$ are connected by both a red and blue path simultaneously.

{{references}} {cite:p}`GraseggerLegersky2024`
:::


## Computations
It is shown that the existence problem of a NAC-coloring for a general graph is NP-complete {cite:p}`Garamvölgyi2022{Thm 3.5}`.
The same is true even for graphs with degree at most 5 {cite:p}`LastovickaLegersky2024{Thm 2.1}`.
Nevertheless, for many reasonably small graphs NAC-colorings can be computed.

In the implementation here we use the strategy from {cite:p}`LastovickaLegersky2024{Sec 3}`:
* Compute all {prf:ref}`NAC-mono classes<def-nac-mono>`
* Decompose the graph into smaller edge-disjoint subgraphs by either
    * taking consecutive NAC-mono classes, or by
    * heuristically clustering NAC-mono classes which are close to each other {cite:p}`LastovickaLegersky2024{Alg 1}`
* Compute all NAC-colorings of these subgraphs
* Merge the NAC-colorings of subgraphs to larger graphs until the full graph is obtained by either
    * joining the next subgraph step by step, or by
    * picking pairs of subgraphs with many common vertices


:::{prf:definition} NAC-mono class
:label: def-nac-mono

Let $G=(V,E)$ be a graph.
An equivalence relation $\sim$ on $E$ is called _NAC-valid_ if
for every NAC-coloring $\delta$ of $G$ we have that $e_1 \sim e_2 \implies \delta(e_1) = \delta(e_2)$
for all edges $e_1,e_2\in E$.

An equivalence class of a NAC-valid relation is called a _NAC-mono class_.

{{references}} {cite:p}`LastovickaLegersky2024{Def 3.2}`
:::

The simplest NAC-mono class is a single edge.
Also triangle-connected components form NAC-mono classes.

:::{prf:definition} triangle-connected component
:label: def-triangle-connected-comp

Let $G=(V,E)$ be a graph.
Let $\triangle$ be the equivalence relation on $E$ given by the reflexive-transitive closure of the relation, where $e_1\triangle e_2$ if
there is a 3-cycle in $G$ containing both $e_1$ and $e_2$.

Clearly, $\triangle$ is NAC-valid.
The equivalence classes are called _triangle-connected components_ or _$\triangle$-components_.

{{references}} {cite:p}`GraseggerLegerskySchicho2019{Def 4.1}`
:::

In the implementation a slightly more specific relation is used.

:::{prf:definition} triangle-extended class
:label: def-triangle-extended-class

Let $G=(V,E)$ be a graph.
Let $\hat \triangle$ be the equivalence relation on $E$ induced by
* $e_1 \triangle e_2$ implies $e_1 \hat \triangle e_2$
* if $e=\{u,v\}$ and there are edges $e_1=\{u,w_1\}, e_2=\{v,w_2\}$ with $e_1\hat\triangle e_2$, then $e\hat\triangle e_2$
* if $e_1=\{u,v_1\}, e_2=\{u,v_2\}$ and there are edges $f_1=\{v_1,w_1\}, f_2=\{v_2,w_2\}$ with $f_1\hat\triangle f_2$, then $e_1\hat\triangle e_2$

Then $\hat\triangle$ is NAC-valid.
The equivalence classes are NAC-mono classes and they are called _triangle-extended classes_ or _$\hat\triangle$-classes_.

{{references}} {cite:p}`LastovickaLegersky2024`
:::

