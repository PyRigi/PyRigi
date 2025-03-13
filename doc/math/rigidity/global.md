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
in $\mathbb{R}^d$, then all {prf:ref}`generic frameworks <def-gen-realization>` $p$ of $G$ are globally $d$-rigid.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

The converse of this theorem is the following one:

:::{prf:theorem}
:label: thm-inverse-k-min-stress-matrix

If a graph $G$ with $d+2$ or more vertices does not have a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then any {prf:ref}`generic framework <def-gen-realization>` $p$ of $G$ is not globally $d$-rigid.

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

The following randomized algorithm from {cite:p}`GortlerHealyThurston2010` checks for global $d$-rigidity.

:::{prf:algorithm}
:label: alg-randomized-globally-rigid
**Input:** A graph $G$ with at least $d + 2$ vertices and a dimension $d$

**Output:** A statement on whether $G$ is globally $d$-rigid, `True` or `False`

Let $n$ be the number of vertices, $m$ be the number of edges,
$t = n\cdot d - \binom{d+1}{2}$ and $N = A\cdot n\cdot \binom{n}{2} +2$, where $A$ is some constant.


1. If $m < t$, output `False` (as the graph cannot even be generically $d$-rigid with so few edges), otherwise continue.
2. Pick a framework with integer coordinates randomly chosen from 1 to $N$.
3. Pick an equilibrium stress vector in a suitably random way. (If $m = t$, there are no stresses, so we consider the zero vector.)
4. Consider the corresponding equilibrium stress matrix and compute its rank.
5. If the rank is $n-d-1$, return `True`, otherwise return `False`.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`

{{references}} {cite:p}`GortlerHealyThurston2010`
:::

The above algorithm may give false negatives as the following theorem tells.
:::{prf:theorem}
:label: thm-globally-randomize-algorithm
The randomized {prf:ref}`algorithm for checking global d-rigidity<alg-randomized-globally-rigid>`  never returns a false positive answer,
and returns a false negative answer with probability bounded above by $nm/N$, where $n$ is the
number of vertices, $m$ is the number of edges and $N$ is an arbitrarily large integer.
In this case, we chose $N \geq A\cdot nm + 2$ so that the probability of getting a false negative
is less than $1/A$.
In particular, checking for generic global $d$-rigidity is in $RP$, i.e.,
the class of randomized polynomial time algorithms.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`

{{references}} {cite:p}`GortlerHealyThurston2010`
:::
