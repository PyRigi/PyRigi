# Graph Theory

Here we introduce graph theoretical concepts related to Rigidity Theory.

## General notions

:::{prf:definition} Union
:label: def-union-graph

Let $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ be simple graphs.
The _union_ of $G_1$ and $G_2$ is the simple graph whose vertex set is $V_1 \cup V_2$
and whose edge set is $E_1 \cup E_2$.

{{pyrigi_crossref}} {meth}`~.Graph.__add__`
:::


:::{prf:definition} t-sum
:label: def-t-sum

Given three graphs $G=(V,E)$, $G_1=(V_1,E_1)$, and $G_2=(V_2,E_2)$, we say that
$G$ is a _$t$-sum_ of $G_1,G_2$ along an edge $e$ if $G=(G_1\cup G_2)-e$,
$G_1\cap G_2=K_t$ and $e\in E_1\cap E_2$.

{{pyrigi_crossref}} {meth}`~.Graph.sum_t`
:::


## Sparse and tight graphs

:::{prf:definition} $(k, \ell)$-sparse and $(k, \ell)$-tight
:label: def-kl-sparse-tight

Let $G = (V, E)$ be a multigraph and let $k, \ell \in \NN$ such that $0\leq \ell < 2k$.
Then $G$ is said to be _$(k, \ell)$-sparse_ if every set of $n'$ vertices spans at most $\max(0,kn' - \ell)$ edges,
or equivalently if every set of $n'$ vertices with at least one edge spans at most $kn' - \ell$ edges.

Let $G = (V, E)$ be a simple graph without loops and let $k, \ell \in \NN$ such that $0\leq \ell \leq \binom{k+1}{2}$.
The graph $G$ is said to be _$(k, \ell)$-sparse_ if every set of $n'$ vertices with $n' \geq k$ spans at most $kn' - \ell$ edges.

A (multi)graph $G$ is said to be _$(k, \ell)$-tight_ if it is $(k, \ell)$-sparse and $|E| = k|V| - \ell$.

{{pyrigi_crossref}} {meth}`~.Graph.is_kl_sparse`
{meth}`~.Graph.is_sparse`
{meth}`~.Graph.is_tight`
{meth}`~.Graph.is_kl_tight`

{{references}} {cite:p}`LeeStreinu2008`, {cite:p}`KiralyMihalyko2022`
:::

:::{prf:lemma}
:label: lem-sparsity-definitions

For simple graphs without loops and with $0\leq \ell < 2k$ the two sparsity definitions from {prf:ref}`def-kl-sparse-tight` are equivalent.
:::

:::{prf:algorithm} Pebble-Game --- Basic Idea
:label: alg-pebble-game
**Input:** A simple graph $G$ (possibly with loops), integers $k>0$ and $\ell$ with $0\leq \ell < 2k$

**Output:** `True` or `False`, whether or not $G$ is $(k,\ell)$-{prf:ref}`sparse<def-kl-sparse-tight>` resp. $(k,\ell)$-tight

1. Start with a new graph $G'$ on the same set of vertices $V$, but no edges.
2. Put $k$ pebbles on every vertex of $G'$.
3. Loop over all edges of $G$. For an edge $e$:
    1. If the vertices of $e$ have together at least $\ell+1$ pebbles:
        * Add a directed edge to $G'$ between these vertices and remove one pebble from its starting vertex.
    2. Else:
        * Pick a vertex $v$ of $e$ with less than $k$ pebbles.
        * Try to find a pebble reachable by a path in $G'$ starting at $v$.
        * If such a path is found, revert all edges in the path, move the pebble to $v$ and go to step 3.1 considering $e$ again.
        * If no such path is found for both vertices of the edge, reject the edge and return `False`.
4. If no edge was rejected there are at least $\ell$ pebbles left.
   For Sparsity return `True`.
   For Tighness return `True` only if there are exactly $\ell$ pebbles left.

{{references}} {cite:p}`JacobsHendrickson1997` {cite:p}`LeeStreinu2008`
:::

:::{prf:definition} Pebble Digraph
:label: def-pebble-digraph
The graph $G'$ after running the {prf:ref}`pebble game algorithm <alg-pebble-game>` is called the _pebble digraph_.

:::

## Graph extensions

:::{prf:definition} k-extension
:label: def-k-extension

Let $d,k \in \NN$.
Let $G=(V,E)$ be a graph, let $F \subset E$ with $|F|=k$
and let $v \notin V$.
Let $H=(W,F)$ be the subgraph of $G$ induced by $F$.
Let further $S \subset V$ be a set of vertices such that
$S \cap W= \emptyset$ and $|S|+|W|=d+k$.
We define
\begin{equation*}
 E_v = \bigl\{ \{v,u\} : u \in W \cup S \bigr\} \,.
\end{equation*}
Then
\begin{equation*}
 G'= \bigl( V \cup \{v\}, (E \setminus F) \cup E_v \bigr)
\end{equation*}
is called a $d$-dimensional _k-extension_ of $G$.

{{pyrigi_crossref}} {meth}`~.Graph.k_extension`
{meth}`~.Graph.one_extension`
{meth}`~.Graph.zero_extension`
{meth}`~.Graph.all_k_extensions`
{meth}`~.Graph.extension_sequence`
:::


:::{prf:definition} 2-tree
:label: def-2-tree

A graph is a _2-tree_ if it can be obtained from a single edge by a sequence of 2-dimensional 0-extensions on adjacent vertices only.

:::


## Connectivity

:::{prf:definition} connected
:label: def-connected

In a graph $G$, two vertices $u$ and $v$ are called
_connected_ if $G$ contains a path from $u$ to $v$.
A graph $G$ is said to be _connected_ if every pair of
vertices in the graph is connected.

:::

:::{prf:definition} k-connected
:label: def-k-connected

Let $k\in\NN$ and $k\geq 1$. A graph $G$ is _$k$-(vertex-)connected_ if it has
at least $k+1$ vertices and it remains {prf:ref}`connected <def-connected>`
when at most $k-1$ vertices are removed. In case $k=2$ it is called
_biconnected_, and in case $k=3$ it is called _triconnected_.

Given a graph $G$, a _$k$-(vertex)-connected component_ of $G$ is a
$k$-connected subgraph of $G$ that is not strictly contained in any
other $k$-connected subgraph of $G$.

:::


:::{prf:definition} local connectivity
:label: def-kappa-G-u-v

Let $G = (V,E)$ be a graph and $u,v\in V$, we use
_$\kappa_G(u,v)$_ to denote the _local connectivity of $u,v$_, which is
the maximum number of pairwise internally disjoint paths from $u$ to $v$ in $G$,
where two paths are internally disjoint if they do not share any edge.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:definition} separating set
:label: def-separating-set

A subset $U\subsetneq V$ is called a _separating set_ of a graph $G=(V,E)$ if
$G-U$ is not {prf:ref}`connected <def-connected>`.

In particular, if $|U| = 2$ then U is called a _separating pair_.

We say that $U$ _separates_ the vertices $u$ and $v$, or that $U$ is a _$(u,v)$-separating set_,
if $u$ and $v$ are in different connected components of $G-U$. 

{{pyrigi_crossref}} {meth}`~.Graph.is_separating_set`
{meth}`~.Graph.is_uv_separating_set`
:::


:::{prf:definition} stable set
:label: def-stable-set

Let $G = (V, E)$ be a graph.
The set $S \subset V$ is called a _stable set_ of $G$
if there is no edge $uv$ in $G$ such that $u,v \in S$.

{{pyrigi_crossref}} {meth}`~.Graph.is_stable_set`
{meth}`~.Graph.is_stable_separating_set`
{meth}`~.Graph.stable_separating_set`
:::


:::{prf:definition} clique
:label: def-clique

A _clique_ of a graph $G$ is an induced subgraph of $G$ that is complete.

:::

## Coning

:::{prf:definition} Cone graph
:label: def-cone-graph

Let $G=(V,E)$ be a graph. _Coning a graph_ adds a new vertex $v^*\notin V$ and adds edges $\{u,v^*\}$
for all vertices $u\in V$ so that $E^*=E\cup \{\{u,v^*\}\,:\, u\in V\}$,
creating the _cone graph_ $G*\{v^*\} = (V\cup \{v^*\}, E^*)$.

{{pyrigi_crossref}} {meth}`~.Graph.cone`
:::


## Apex Graphs

:::{prf:definition} Apex graphs
:label: def-apex-graph

Let $G=(V,E)$ be a graph and let $k$ be an integer. If the removal of some set of $k$ vertices from $G$
results in a planar graph, we call $G$ a _$k$-apex graph_ or _$k$-vertex apex graph_. Similarly, if the removal of some set
of $k$ edges from $G$ results in a planar graph, we call $G$ a _$k$-edge apex graph_.

Moreover, if one of these properties holds for all choices of $k$ vertices or edges, we call the graph a
_critically $k$-vertex apex graph_ or _critically $k$-edge apex graph_, respectively.

{{pyrigi_crossref}} {meth}`~.Graph.is_vertex_apex`
{meth}`~.Graph.is_k_vertex_apex`
{meth}`~.Graph.is_edge_apex`
{meth}`~.Graph.is_k_edge_apex`
{meth}`~.Graph.is_critically_vertex_apex`
{meth}`~.Graph.is_critically_k_vertex_apex`
{meth}`~.Graph.is_critically_edge_apex`
{meth}`~.Graph.is_critically_k_edge_apex`
:::


:::{toctree}
:maxdepth: 2
graph-theory/examples
