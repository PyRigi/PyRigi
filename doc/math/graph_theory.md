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


## Generalized double banana

:::{prf:definition} Generalized double banana
:label: def-generalized-double-banana

For $d\geq 3$ and $2\leq t\leq d-1$, 
the graph $B_{d,t}$ is defined by putting $B_{d,t}=(G_1\cup G_2)-e$ 
where $G_i\cong K_{d+2}$, $G_1\cap G_2\cong K_{t}$ and $e\in E(G_1\cap G_2)$. 
Note that the graph $B_{3,2}$  is the well known flexible $\mathcal{R}_3$-circuit,
commonly referred to as the _double banana_.

{{pyrigi_crossref}} {meth}`~pyrigi.graphDB.DoubleBanana`
:::

The family  $\mathcal{B}_{d,d-1}^+$ consists of all graphs of the form
$(G_1\cup G_2)-\{e,f,g\}$ where: $G_1\cong K_{d+3}$ and $e,f,g\in E(G_1)$;
$G_2\cong K_{d+2}$ and $e\in E(G_2)$; $G_1\cap G_2\cong K_{d-1}$;
$e,f,g$ do not all have a common end-vertex;
if $\{f,g\}\subset E(G_1)\setminus E(G_2)$ then $f,g$ do not have a common end-vertex.


:::{prf:theorem}
:label: thm-flexible-circuit-classification

Suppose $G$ is a flexible $\mathcal{R}_d$-circuit with at most $d+6$ vertices. Then either

* $d=3$ and $G\in \{B_{3,2}\}\cup \mathcal{B}_{3,2}^+$ or
* $d\geq 4$ and $G\in \{B_{d,d-1}$, $B_{d,d-2}\}\cup \mathcal{B}_{d,d-1}^+$.

{cite:p}`Grasegger2022`
:::


:::{prf:definition} 2-sum
:label: def-2-sum

Given three graphs $G=(V,E)$, $G_1=(V_1,E_1)$, and $G_2=(V_2,E_2)$, we say that  
$G$ is a _$2$-sum_ of $G_1,G_2$ along an edge $e$ if $G=(G_1\cup G_2)-e$, 
$G_1\cap G_2=K_2$ and $e\in E_1\cap E_2$.

{{pyrigi_crossref}} {meth}`~pyrigi.graphDB.sum_2`
:::