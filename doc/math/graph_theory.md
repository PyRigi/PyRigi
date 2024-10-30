# Graph Theory

Here we introduce graph theoretical concepts related to Rigidity Theory. 

## Sparse and tight graphs

:::{prf:definition} $(k, \ell)$-sparse and $(k, \ell)$-tight
:label: def-kl-sparse-tight

Let $G = (V, E)$ be a (multi)graph and let $k, \ell \in \NN$.
The graph $G$ is said to be _$(k, \ell)$-sparse_ if every set of $n'$ vertices with $k\leq n' \leq |V|$ spans at most $kn' - \ell$ edges.
The graph $G$ is said to be _$(k, \ell)$-tight_ if it is $(k, \ell)$-sparse and $k|V| - \ell = |E|$.

{{pyrigi_crossref}} {meth}`~.Graph.is_sparse`
{meth}`~.Graph.is_tight`

{{references}} {cite:p}`Lee2008`
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