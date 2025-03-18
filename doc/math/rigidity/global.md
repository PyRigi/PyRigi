# Global Rigidity

:::{prf:definition}
:label: def-globally-rigid-graph

A graph $G = (V,E)$ is called _globally $d$-rigid_,
if for every {prf:ref}`generic <def-gen-realization>` $d$-dimensional {prf:ref}`framework <def-framework>` $(G,p)$,
all $d$-dimensional frameworks $(G,p')$ {prf:ref}`equivalent <def-equivalent-framework>` to $(G,p)$
are {prf:ref}`congruent <def-equivalent-framework>` to $(G,p)$.

{{references}} {cite:p}`JacksonJordan2005`

{meth}`~.Graph.is_globally_rigid`
:::


:::{prf:theorem}
:label: thm-globally-redundant-3connected

A graph $G$ is globally $2$-rigid if and only if it either is a
complete graph on at most three vertices or it is $3$-connected and {prf:ref}`redundantly rigid<def-redundantly-rigid-graph>`.

{{references}} {cite:p}`JacksonJordan2005{Thm 7.1}`
:::

:::{prf:theorem}
:label: thm-globally-mindeg6-dim2

Let $G$ be a $6$-connected {prf:ref}`2-rigid <def-gen-rigid>` graph. Then $G$ is globally $2$-rigid.

{{references}} {cite:p}`JacksonJordan2005{Thm 7.2}`
:::

:::{prf:definition}
:label: def-stress-kernel

Let $G$ be a graph, if $\Omega$ is an {prf:ref}`equilibrium stress matrix <def-stress-matrix>`,
its kernel is called _stress kernel_; we denote it by $K(\Omega)$ and its dimension by $k(\Omega)$.
We denote by $k_{min}(G,d)$ the minimal value of $k(\Omega)$ as $\Omega$ ranges over all
{prf:ref}`equilibrium stress matrices <def-stress-matrix>` of all
{prf:ref}`generic d-dimensional frameworks <def-gen-realization>` of $G$.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:lemma}
:label: lem-k-min-stress-matrix


For {prf:ref}`frameworks <def-framework>` of a graph $G$ with at least $d+1$ vertices,
the relation $k_{min}(G,d) \geq d+1$ holds.

{{references}} {cite:p}`GortlerHealyThurston2010{Lem 1.11}`
:::

:::{prf:definition}
:label: def-has-min-stress-kernel

A graph $G$ is said to have
a _minimal {prf:ref}`stress kernel <def-stress-kernel>` in $\RR^d$_
if $k_{min}(G,d) = d+1$.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:theorem}
:label: thm-k-min-stress-matrix

Let $G$ be a graph with $d+2$ or more vertices. If $G$ has a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\RR^d$, then all {prf:ref}`generic frameworks <def-gen-realization>` $(G,p)$ are globally $d$-rigid.

{{references}} {cite:p}`GortlerHealyThurston2010{Thm 1.13}`
:::

The converse of this theorem is the following one:

:::{prf:theorem}
:label: thm-inverse-k-min-stress-matrix

Let $G$ be a graph with $d+2$ or more vertices. If $G$ does not have a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\RR^d$, then any $d$-{prf:ref}`dimensional generic framework <def-gen-realization>` $(G,p)$ is not globally $d$-rigid.

{{references}} {cite:p}`GortlerHealyThurston2010{Thm 1.14}`
:::

The following randomized algorithm from {cite:p}`GortlerHealyThurston2010` checks for global $d$-rigidity.

:::{prf:algorithm}
:label: alg-randomized-globally-rigid
**Input:** A natural number $d$ and a graph $G = (V, E)$ with at least $d + 2$ vertices

**Output:** `True` or `False`, whether or not $G$ is globally $d$-rigid

Let $n = |V|$ and $m = |E|$, let
$t = n\cdot d - \binom{d+1}{2}$, and $N = A \cdot n\cdot \binom{n}{2} +2$, where $A$ is some constant.

1. If $m < t$, output `False` (as the graph cannot even be generically $d$-rigid with so few edges), otherwise continue.
2. Pick a framework with integer coordinates randomly chosen from 1 to $N$.
3. Pick an equilibrium stress vector in a suitably random way. (If $m = t$, there are no stresses, so we consider the zero vector.)
4. Consider the corresponding equilibrium stress matrix and compute its rank.
5. If the rank is $n-d-1$ then return `True`, otherwise return `False`.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

The above algorithm may give false negatives as the following theorem tells.

:::{prf:theorem}
:label: thm-globally-randomize-algorithm
The randomized {prf:ref}`algorithm for checking global d-rigidity<alg-randomized-globally-rigid>`  never returns a false positive answer,
and returns a false negative answer with probability bounded above by $nm/N$, where $n$ is the
number of vertices, $m$ is the number of edges and $N$ is an arbitrarily large integer.
Given a number $A$, we chose $N \geq A\cdot nm + 2$
so that the probability of getting a false negative
is less than $1/A$.
In particular, checking for generic global $d$-rigidity is in $RP$, i.e.,
the class of randomized polynomial time algorithms.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:definition} globally linked in a framework
:label: def-globally-linked-p

We say that a pair of vertices $\{u,v\}$ is _globally linked
in a $d$-{prf:ref}`dimensional framework <def-framework>` $(G,p)$_
if for every {prf:ref}`equivalent <def-equivalent-framework>`
$d$-{prf:ref}`dimensional framework <def-framework>` $(G,q)$ we have
$||p(u)-p(v)|| = ||q(u)-q(v)||$. This is not a generic property.

{{references}} {cite:p}`JordanVillanyi2024`
:::

:::{prf:definition} globally linked in a graph
:label: def-globally-linked

A pair of vertices $\{u,v\}$ is _globally $d$-linked in $G$_ if it is
{prf:ref}`globally linked <def-globally-linked-p>` in all $d$-dimensional
{prf:ref}`generic frameworks <def-gen-realization>` $(G,p)$.

A pair $\{u,v\}$ is _weakly globally $d$-linked in $G$_ if there exists
a $d$-dimensional {prf:ref}`generic framework <def-gen-realization>` $(G,p)$ in which $\{u,v\}$
is {prf:ref}`globally linked <def-globally-linked-p>`.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:theorem}
:label: thm-weakly-globally-linked-globally-rigid-graph

A graph $G$ is {prf:ref}`globally d-rigid <def-globally-rigid-graph>` if and only if every pair of
vertices are {prf:ref}`weakly globally d-linked <def-globally-linked>` in $G$.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:corollary}
:label: cor-weakly-globally-linked-rigid-graph

Given a $d$-rigid but not globally $d$-rigid graph $G$,
there exists at least one pair of vertices of $G$
that are not {prf:ref}`weakly globally d-linked <def-globally-linked>` in $G$.
:::


:::{prf:definition} linked pair
:label: def-linked-pair

A pair of vertices $\{u,v\}$ of $G$ is _linked in a
$d$-{prf:ref}`dimensional framework <def-framework>` $(G,p)$_
(or that $uv$ is an _implied edge_ of $(G,p)$) if the set of distances
$\{ \|q(u) - q(v) \| : (G,q) \text{ is equivalent to } (G,p) \}$
is finite.

A pair of vertices $\{u,v\}$ of $G$ is _$d$-linked_ if it is linked
in all $d$-dimensional {prf:ref}`generic frameworks <def-gen-realization>` $(G,p)$.

{{references}} {cite:p}`Jordan2016`
:::


:::{prf:lemma}
:label: lem-linked-pair-rigid-component

A pair $\{u, v\}$ is 2-{prf:ref}`linked <def-linked-pair>` in $G$ if and only if
there exists a 2-{prf:ref}`rigid component <def-rigid-components>` of $G$ containing
$u$ and $v$.

{{references}} {cite:p}`Jordan2016`
:::


:::{prf:lemma}
:label: lem-linked-pair-r2-circuit

A pair $\{u, v\}$ of vertices is non-adjacent and {prf:ref}`2-linked <def-globally-linked>`
in $G$ if and only if there exists some subgraph $G_0 = (V_0,E_0)$ of $G$ with $u,v\in V_0$
such that $G_0+uv$ is an $\mathcal{R}_2$-{prf:ref}`fundamental circuit <def-fundamental-circuit>`.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:definition} augmented graph
:label: def-augmented-graph

Given a graph $G$, its _augmented graph_ is the graph obtained from $G$
by adding an edge between every {prf:ref}`separating pair <def-separating-set>` of $G$.
:::


:::{prf:definition} cleaving operation
:label: def-cleaving-operation

Let $G=(V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>`
and $u,v\in V$ be such that $\{u,v\}$
is a {prf:ref}`separating pair <def-separating-set>` of $G$.
Let $C$ be a {prf:ref}`connected component <def-k-connected>`
of $G-\{u,v\}$ and let $H$ be the subgraph of $G$ induced by $V(C) \cup \{u,v\}$.
We say that $H + uv$ is obtained from $G$ by a _cleaving operation_ along $\{u,v\}$.
:::


:::{prf:lemma}
:label: lem-3-block

Let $G=(V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` and let $\{u,v\}$ be a
non-adjacent vertex pair in $G$ with {prf:ref}`local connectivity <def-kappa-G-u-v>` $\kappa_G(u,v) \geq 3$.
Then either $\{u,v\}$ is a {prf:ref}`separating pair <def-separating-set>` in $G$ or there is
a unique {prf:ref}`3-connected component <def-k-connected>` $B$ of the
{prf:ref}`augmented graph <def-augmented-graph>` of $G$ such that $\{u,v\} \subset V(B)$.
In the latter case the subgraph $B$ can be obtained from $G$ by a sequence of
{prf:ref}`cleaving operations <def-cleaving-operation>`.
Furthermore, $uv \notin E(B)$ and if the pair $\{u,v\}$ is 2-linked in $G$ then it is also
2-linked in $B$.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:definition} 3-block
:label: def-block-3

Let $G=(V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` and let $\{u,v\}$ be a
non-adjacent vertex pair in $G$ which is not a {prf:ref}`separating pair <def-separating-set>`.
The unique {prf:ref}`3-connected component <def-k-connected>`
$B$ of the {prf:ref}`augmented graph <def-augmented-graph>` of $G$ such that $\{u,v\}\subset V(B)$
is called the _3-block_ of $\{u,v\}$ in $G$.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:theorem}
:label: thm-weakly-globally-linked

Let $G = (V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` and let $\{u,v\}$ be
a non-adjacent 2-linked pair of vertices with {prf:ref}`local connectivity <def-kappa-G-u-v>` $\kappa_G(u,v) \geq 3$.
Then $\{u,v\}$ is {prf:ref}`weakly globally 2-linked <def-globally-linked>` in $G$ if and only if either
* $\{u,v\}$ is a {prf:ref}`separating pair <def-separating-set>` in G, or
* $C(B,V_0)$ is {prf:ref}`globally 2-rigid <def-globally-rigid-graph>`,

where $B$ is the {prf:ref}`3-block <def-block-3>` of $\{u,v\}$ in $G$, $B_0 = (V_0,E_0)$
is a subgraph of $B$ with $u,v \in V_0$ such that $B_0 + uv$ is an $\mathcal{R}_2$-{prf:ref}`circuit <def-matroid>`,
and $C(B, V_0)$ is the graph obtained as follows.

Let $V_1,\dots, V_r$ be the vertex sets of the {prf:ref}`connected components <def-k-connected>` of $B-V_0$.
Delete from $B$ the vertex sets $V_i$ for $1\leq i\leq r$
and add the edges $xy$ for all pairs $x,y \in N_B(V_i)$ for $1\leq i\leq r$.
Here $N_B(V_i)$ denotes the set of nodes of $B-V_i$ that are connected by an edge to some vertex of $V_i$.
The resulting graph is $C(B, V_0)$.

{{pyrigi_crossref}} {meth}`~.Graph.is_weakly_globally_linked`

{{references}} {cite:p}`JordanVillanyi2024{Thm 5.8}`
:::
