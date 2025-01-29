# Matroid Theory

Here we introduce matroidal concepts related to Rigidity Theory.

## General Matroid Theory

:::{prf:definition} Matroid
:label: def-matroid

A _matroid_ $\mathcal{M}=(E, \mathcal{I})$ is a pair consisting of a finite set $E$ (called the _ground set_) and $\mathcal{I}$ is a family of subsets of $E$ (called the _independent sets_) with the following properties:

* $\emptyset \in \mathcal{I}$;
* for every $A \subset B \subset E$, if $B \in \mathcal{I}$ then $A \in \mathcal{I}$; and
* if $A,B\in \mathcal{I}$ and $|A|>|B|$ then there exists $x\in A\setminus B$ such that $B\cup \{x\}\in \mathcal{I}$.

A subset of the ground set $E$ that is not independent is called _dependent_. A maximal independent set
– that is, an independent set that becomes dependent upon adding any element of $E$ –
is called a _basis_ for the matroid.
A _circuit_ is a minimal dependent subset of $E$
– that is, a dependent set whose proper subsets are all independent.

{{pyrigi_crossref}} {meth}`~.Graph.is_Rd_independent`
{meth}`~.Graph.is_Rd_dependent`
{meth}`~.Graph.is_Rd_circuit`
:::


:::{prf:definition} Rank function and closure
:label: def-rank-function-closure

The _rank function_ $r$ of a {prf:ref}`matroid <def-matroid>` $\mathcal{M}$
is the function $r \colon E \rightarrow \NN$
such that $r(A)=\max\{|I| \colon I\subset A, I \in \mathcal{I} \}$.
The rank function has the following properties:

* for any $A\subset E$, $0\leq r(A) \leq |A|$;
* any two subsets $A,B\subset E$, $r(A\cap B) + r(A\cup B) \leq r(A)+r(B)$; and
* any $A\subset E$ and any $x\notin A$, $r(A) \leq r(A\cup \{x\}) \leq r(A)+1$.

For $A\subset E$, the _closure_ of $A$ is the set $\textrm{cl}(A)=\{x\in E: r(A)=r(A\cup \{x\}) \}$. We call $A\subset E$ _closed_ if $A = \textrm{cl}(A)$.

{{pyrigi_crossref}} {meth}`~.Graph.is_Rd_closed`
:::


:::{prf:definition} Coloop
:label: def-coloop

Let $\mathcal{M}=(E, \mathcal{I})$ be a matroid. An element in $E$
that belongs to no {prf:ref}`circuit <def-matroid>` is called a _coloop_.
Equivalently, an element is a coloop if it belongs to every {prf:ref}`basis <def-matroid>`.

:::


:::{prf:definition} $k$-fold circuit
:label: def-k-circuit

Let $\mathcal{M}=(E,r)$ be a matroid with finite ground set $E$ and rank function $r$.
An equivalent definition of a _circuit_ of $\mathcal{M}$ is a set $C\subseteq E$ such that $r(C)=|C|-1=r(C-e)$
for all $e\in E$. This generalizes to _$k$-fold circuits_ in $\mathcal{M}$, which are given by sets $D\subseteq E$ such that
$r(D)=|D|-2=r(D-e)$ for all $e\in D$, for some fixed integer $k\geq 0$.

{{references}} {cite:p}`JacksonNixonSmith2024`
:::

## Rigidity Matroid

:::{prf:definition} Rigidity matroid
:label: def-rigidity-matroid
The _$d$-dimensional rigidity matroid_ of a {prf:ref}`framework <def-framework>` $(G, p)$ in $\mathbb{R}^d$ is the row matroid of the {prf:ref}`rigidity matrix <def-rigidity-matrix>` $R_d(G,p)$. That is, a set $F\subseteq E$ is independent whenever the corresponding rows of $R_d(G,p)$ are linearly independent.
:::


:::{prf:definition} Generic rigidity matroid
:label: def-gen-rigidity-matroid
The _generic $d$-dimensional rigidity matroid_ of a graph $G=(V,E)$ is the {prf:ref}`matroid <def-matroid>` $\mathcal{R}_d(G)$ on $E$ in which a set of edges $F\subseteq E$ is independent whenever the corresponding rows of $R_d(G,p)$ are independent, for some (or equivalently every) {prf:ref}`generic realization <def-gen-realization>` $p$ of $G$.
:::

:::{prf:lemma}
:label: lem-2-sum

Suppose that $G=(V,E)$ is the {prf:ref}`2-sum <def-t-sum>` of $G_1=(V_1,E_1)$ and $G_2=(V_2,E_2)$.
Then $G$ is an $\mathcal{R}_d$-circuit if and only if $G_1$ and $G_2$ are both
$\mathcal{R}_{d}$-circuits.

{{references}} {cite:p}`GraseggerGulerEtAl2022`
:::


:::{prf:lemma}
:label: lem-k-sum

Let $k\geq 1$ be an integer and let $G$ be the graphical 2-sum of two graphs $G_1$ and
$G_2$ along an edge $e$.
Suppose that $e$ is not a {prf:ref}`coloop <def-coloop>` in either $\mathcal{R}_d(G_1)$ or $\mathcal{R}_d(G_2)$.
Then $G$ is a $k$-fold circuit in $\mathcal{R}_d$ if and only if $G_1$ is a $k_1$-fold
$\mathcal{R}_d$-circuit and $G_2$ is a $k_2$-fold $\mathcal{R}_d$-circuit for some
$k_1,k_2\geq 1$ with $k_1+k_2=k+1$.

{{references}} {cite:p}`JacksonNixonSmith2024`
:::