# Matroid Theory

Here we introduce matroidal concepts related to Rigidity Theory.

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

For $A\subset E$, the _closure_ of $A$ is the set $\textrm{cl}(A)=\{x\in E: r(A)=r(A\cup \{x\}) \}$.
:::


:::{prf:definition} Rigidity matroid
:label: def-rigidity-matroid
The _$d$-dimensional rigidity matroid_ of a {prf:ref}`framework <def-framework>` $(G, p)$ in $\mathbb{R}^d$ is the row matroid of the {prf:ref}`rigidity matrix <def-rigidity-matrix>` $R_d(G,p)$. That is, a set $F\subseteq E$ is independent whenever the corresponding rows of $R_d(G,p)$ are linearly independent.
:::


:::{prf:definition} Generic rigidity matroid
:label: def-gen-rigidity-matroid
The _generic $d$-dimensional rigidity matroid_ of a graph $G=(V,E)$ is the {prf:ref}`matroid <def-matroid>` $\mathcal{R}_d(G)$ on $E$ in which a set of edges $F\subseteq E$ is independent whenever the corresponding rows of $R_d(G,p)$ are independent, for some (or equivalently every) {prf:ref}`generic realization <def-gen-realization>` $p$ of $G$.
:::
