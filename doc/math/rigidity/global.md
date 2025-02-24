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
it holds $k_{min}(G,d) \geq d+1$.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:definition}
:label: def-has-min-stress-kernel

A graph $G$ has a _minimal {prf:ref}`stress kernel <def-stress-kernel>` in $\mathbb{R}^d$_
if $k_{min}(G,d) = d+1$.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:theorem}
:label: thm-k-min-stress-matrix

If a graph $G$ with $d+2$ or more vertices has a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then all {prf:ref}`generic frameworks <def-gen-realization>` $p$ of $G$ are globally rigid.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

The converse of this theorem is the following one:

:::{prf:theorem}
:label: thm-inverse-k-min-stress-matrix

If a graph $G$ with $d+2$ or more vertices does not have a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then any {prf:ref}`generic framework <def-gen-realization>` $p$ of $G$ is not globally rigid.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::
The method {{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid` uses the following randomized algorithm:

Let $d$ be the dimension for which we want to test whether the graph is globally $d$-rigid,
$v$ be the number of vertices, $e$ be the number of edges,
$t = v\cdot d - \binom{d+1}{2}$ and $N = A\cdot v\cdot \binom{v}{2} +2$, where $A$ is a constant.
To check if a graph with at least $d + 2$ vertices is generically globally rigid in $\RR^d$,
proceed as follows:
* If $e < t$, output `False` (as the graph cannot even be generically locally rigid with so few edges), otherwise continue.
* Pick a framework with integer coordinates randomly chosen from 1 to $N$.
* Pick one equilibrium stress vector in a suitably random way. (If $e = t$, there are no stresses, so we consider the zero vector.)
* Consider the corresponding equilibrium stress matrix and compute its rank.
* If the rank is $v-d-1$, return `True`, otherwise return `False` .

:::{prf:theorem}
:label: thm-globally-randomize-algorithm

The randomized algorithm for checking global rigidity never returns a false `True` answer,
and returns a false `False` answer with probability bounded above by $ve/N$, where $v$ is the
number of vertices, $e$ is the number of edges and $N$ is an arbitrarily large integer.
In this case, we chose $N \geq A\cdot ve + 2$ so that the probability of getting a false `False`
is less than $1/A$.
In particular, checking for generic global rigidity in $\mathbb{R}^d$ is in $RP$, i.e.,
the class of randomized polynomial time algorithms.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`
{{references}} {cite:p}`GortlerHealyThurston2010`
:::

:::{prf:definition} globally linked in a framework
:label: def-globally-linked-p

We say that a pair of vertices $\{u,v\}$ in a $d$-{prf:ref}`dimensional framework <def-framework>`
$(G,p)$ is _globally $d$-linked in $(G,p)$_ if for every {prf:ref}`equivalent <def-equivalent-framework>`
$d$-{prf:ref}`dimensional framework <def-framework>` $(G,q)$ we have
$||p(u)-p(v)|| = ||q(u)-q(v)||$. This is not a generic property.

{{references}} {cite:p}`JordanVillanyi2024`
:::

:::{prf:definition} globally linked in a graph
:label: def-globally-linked

A pair of vertices $\{u,v\}$ is _globally linked in $G$_ in $\RR^d$ if it is
{prf:ref}`globally linked <def-globally-linked-p>` in all $d$-dimensional
{prf:ref}`generic frameworks <def-gen-realization>` $(G,p)$.

A pair $\{u,v\}$ is _weakly globally linked in $G$_ in $\RR^d$ (or _weakly globally $d$-linked_) if there exists
a $d$-dimensional {prf:ref}`generic framework <def-gen-realization>` $(G,p)$ in which $\{u,v\}$
is {prf:ref}`globally linked <def-globally-linked-p>`.

If $\{u,v\}$ is not weakly globally $d$-linked in $G$, then it is called _globally loose in $G$_.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:theorem}
:label: thm-weakly-globally-linked-globally-rigid-graph

A graph $G$ is {prf:ref}`globally rigid <def-globally-rigid-graph>` in $\RR^d$ if and only if every pair of
vertices are {prf:ref}`weakly globally d-linked <def-globally-linked>` in $G$.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:corollary}
:label: cor-weakly-globally-linked-rigid-graph

Given a rigid but not globally rigid graph $G$ in $\RR^d$ then there exists at least one pair of vertices
of $G$ that are not {prf:ref}`weakly globally d-linked <def-globally-linked>` in $G$.

:::


:::{prf:definition}
:label: def-rigid-component

Given a graph $G$, maximal rigid subgraph of $G$ is a _rigid component_ of $G$.
{{references}} {cite:p}`BergJordan2003`
:::

Clearly, every edge belongs to some rigid component and rigid components are induced subgraphs.

:::{prf:lemma}
:label: lem-linked-pair-rigid-component

A pair $\{u, v\}$ is {prf:ref}`linked <def-globally-linked>` in $G$ if and only if
there exists a {prf:ref}`rigid component <def-rigid-component>` of $G$ containing 
$u$ and $v$.

{{references}} {cite:p}`Jordan2016`
:::


:::{prf:lemma}
:label: lem-linked-pair-r2-circuit

A pair $\{u, v\}$ of vertices is non-adjacent and {prf:ref}`linked <def-globally-linked>`
in $G$ if and only if there exists some subgraph $G_0 = (V_0,E_0)$ of $G$ with $u,v\in V_0$
such that $G_0+uv$ is an $\mathcal{R}_2$-{prf:ref}`circuit <def-fundamental-circuit>`.

{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:definition} augmented graph
:label: def-augmented-graph

Given a graph $G$ its _augmented graph_ is the graph obtained from $G$
by adding an edge between every {prf:ref}`separating pair <def-separating-set>` of $G$.

:::


:::{prf:definition} cleaving operation
:label: def-cleaving-operation

Let $G=(V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` and $u,v\in V$ such that $\{u,v\}$
is a {prf:ref}`separating pair <def-separating-set>` of $G$. Let $C$ be a {prf:ref}`connected component <def-k-connected>`
of $G-\{u,v\}$ and let $H$ be the subgraph of $G$ induced by $V(C)\cup \{u,v\}$. Then we say that $H+ uv$
is obtained from $G$ by a _cleaving operation_ along $\{u,v\}$.

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
Furthermore, $uv \notin E(B)$ and, in $\RR^2$, if the pair $\{u,v\}$ is linked in $G$ then it is also
linked in $B$.

{{pyrigi_crossref}} {meth}`~.Graph.block_3`
{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:definition} 3-block
:label: def-block-3

Let $G=(V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` and let $\{u,v\}$ be a
non-adjacent vertex pair in $G$ which is not a {prf:ref}`separating pair <def-separating-set>`.
The unique {prf:ref}`3-connected component <def-k-connected>`
$B$ of the {prf:ref}`augmented graph <def-augmented-graph>` of $G$ such that $\{u,v\}\subset V(B)$
is called the _3-block_ of $\{u,v\}$ in $G$.

{{pyrigi_crossref}} {meth}`~.Graph.block_3`
{{references}} {cite:p}`JordanVillanyi2024`
:::


:::{prf:theorem}
:label: thm-weakly-globally-linked

Let $G = (V,E)$ be a {prf:ref}`2-connected graph <def-k-connected>` in $\RR^2$ and let $\{u,v\}$ be
a non-adjacent linked pair of vertices with {prf:ref}`local connectivity <def-kappa-G-u-v>` $\kappa_G(u,v) \geq 3$.
Then $\{u,v\}$ is {prf:ref}`weakly globally 2-linked <def-globally-linked>` in $G$ if and only if either
* $\{u,v\}$ is a {prf:ref}`separating pair <def-separating-set>` in G, or
* $C(B,V_0)$ is {prf:ref}`globally rigid <def-globally-rigid-graph>`,

where $B$ is the {prf:ref}`3-block <def-block-3>` of $\{u,v\}$ in $G$, $B_0 = (V_0,E_0)$
is a subgraph of $B$ with $u,v \in V_0$ such that $B_0 + uv$ is an $\mathcal{R}_2$-{prf:ref}`circuit <def-matroid>`, and $C(B, V_0)$ is the graph obtained as follows.

Let $V_1,\dots, V_r$ be the vertex sets of the {prf:ref}`connected components <def-k-connected>` of $B-V_0$. Delete from $B$ the vertex
sets $V_i$ for $1\leq i\leq r$ and add the edges $xy$ for all pairs
$x,y \in N_B(V_i)$ for $1\leq i\leq r$. Here $N_B(V_i)$ denotes
the set of nodes of $B-V_i$ that are connected by an edge to some vertex of $V_i$. The resulting graph is $C(B, eV_0)$.

{{pyrigi_crossref}} {meth}`~.Graph.is_weakly_globally_linked`
{{references}} {cite:p}`JordanVillanyi2024`
:::
