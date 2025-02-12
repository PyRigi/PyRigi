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
:::


## Sparse and tight graphs

:::{prf:definition} $(k, \ell)$-sparse and $(k, \ell)$-tight
:label: def-kl-sparse-tight

Let $G = (V, E)$ be a (multi)graph and let $k, \ell \in \NN$.
The graph $G$ is said to be _$(k, \ell)$-sparse_ if every set of $n'$ vertices with $k\leq n' \leq |V|$ spans at most $kn' - \ell$ edges.
The graph $G$ is said to be _$(k, \ell)$-tight_ if it is $(k, \ell)$-sparse and $k|V| - \ell = |E|$.

{{pyrigi_crossref}} {meth}`~.Graph.is_kl_sparse`
{meth}`~.Graph.is_sparse`
{meth}`~.Graph.is_tight`
{meth}`~.Graph.is_kl_tight`

{{references}} {cite:p}`LeeStreinu2008`
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

A graph $G$ is said _$k$-(vertex-)connected_ if it has 
more than $k$ vertices and it remains `connected <def-connected>` 
when less than $k$ vertices are removed. In case $k=2$, it is called 
_biconnected_ and in case $k=3$, it is called _triconnected_.

Given a $k$-connected graph, it decomposes into a tree of 
_$(k+1)$-vertex-connected components_.
:::


:::{prf:definition} kappa-G-u-v
:label: def-kappa-G-u-v

Let $G = (V,E)$ be a graph and $u,v\in V$, we use 
_$\kappa_G(u,v)$_ to denote the _path number of $u,v$_, which is 
the maximum number of pairwise openly (or internally) disjoint $uv$-paths in $G$, 
where two paths are openly disjoint if they do not share any edge.

{{references}} {cite:p}`Jackson2006`
:::


:::{prf:definition} separation-point
:label: def-separation-point

Given a `connected <def-connected>` graph, if there is a distinct 
triple $v,w,a\in V$ such that $a$ is on every path from $v$ to $w$, 
then $a$ is called a _separation point_ of $G$.

:::


:::{prf:definition} separating-pair
:label: def-separating-pair

Let $\{u,v\}$ be a pair of vertices in a 
{prf:ref}`2-connected graph <def-k-connected>` $G$.
They are called a _separating pair_ (or _separation pair_ or _2-separator_) if and 
only if $G-\{u,v\}$ is not `connected <def-connected>`.

{{pyrigi_crossref}} {meth}`~.Graph.is_separating_pair`
:::


:::{prf:definition} clique
:label: def-clique 

A _clique_ is a subset of vertices of a graph such that every two distinct 
vertices in the clique are adjacent. That is, a clique of a graph $G$ is an
induced subgraph of $G$ that is complete.

:::


:::{prf:definition} clique-operation
:label: def-clique-operation

Let $G=(V,E)$ be a graph, $\emptyset \neq X \subseteq V$, and let 
$V_1,\dots, V_r$ be the vertex sets of the `connected components <def-k-connected>` 
of $G-X$. 
The graph _make Clique$(G,X)$ operation_ consists of deleting the vertex 
sets $V_i, 1\leq i\leq r$ and adding a new edge $xy$ for all pairs 
$x,y\in N_G(V_i), xy\notin E$ for $1\leq i\leq r$. Here $N_G(V_i)$ denotes 
the set of nodes of $G-V_i$ that are connected by an edge to at least one 
vertex of $V_i$ (and then to $v_i$, by its definition).

{{pyrigi_crossref}} {meth}`~.Graph.make_outside_neighbors_clique`
{{references}} {cite:p}`Jackson2006`
:::

:::{toctree}
:maxdepth: 2
graph-theory/examples
