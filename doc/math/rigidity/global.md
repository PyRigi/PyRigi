# Global Rigidity

:::{prf:definition}
:label: def-globally-rigid-graph

A graph $G = (V,E)$ is called _globally $d$-rigid_,
if for every {prf:ref}`generic <def-gen-realization>` $d$-dimensional {prf:ref}`framework <def-framework>` $(G,p)$,
all $d$-dimensional frameworks $(G,p')$ {prf:ref}`equivalent <def-equivalent-framework>` to $(G,p)$
are {prf:ref}`congruent <def-equivalent-framework>` to $(G,p)$.

{{references}} {cite:p}`Jackson2005`
:::


:::{prf:theorem}
:label: thm-globally-redundant-3connected

A graph $G$ is globally $2$-rigid if and only if it either is a
complete graph on at most three vertices or it is $3$-connected and {prf:ref}`redundantly rigid<def-redundantly-rigid-graph>`.

{{references}} {cite:p}`Jackson2005{Thm 7.1}`
:::

:::{prf:theorem}
:label: thm-globally-mindeg6-dim2

Let $G$ be a $6$-connected {prf:ref}`2-rigid <def-gen-rigid>` graph. Then $G$ is globally $2$-rigid.

{{references}} {cite:p}`Jackson2005{Thm 7.2}`
:::

:::{prf:definition}
:label: def-stress-kernel

If $\Omega$ is an {prf:ref}`equilibrium stress matrix <def-stress-matrix>`, its kernel is called _stress kernel_;
we denote it by $K(\Omega)$ and its dimension by $k(\Omega)$ or simply $k$.
We denote by $k_{min}(G,d)$ or just $k_{min}$ the minimal value of $k(\Omega)$ 
as $\Omega$ ranges over all {prf:ref}`equilibrium stress matrices <def-stress-matrix>` of all generic 
{prf:ref}`frameworks <def-framework>` in $C^d(G)$, the space of frameworks.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:lemma}
:label: lem-k-min

For {prf:ref}`frameworks <def-framework>` of a graph $G$ with at least $d+1$ vertices, 
$k_{min}(G,d) \geq d+1$.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:definition}
:label: def-has-min-stress-kernel

A graph $G$ _has a minimal {prf:ref}`stress kernel <def-stress-kernel>` in $\mathbb{R}^d$_ 
if $k_{min}(G,d) = d+1$.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:theorem}
:label: thm-k-min

If a graph $G$ with $d+2$ or more vertices has a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then all generic {prf:ref}`frameworks <def-framework>` $p\in C^d(G)$ are globally rigid.

{{references}} {cite:p}`Gortler2010`
:::

The converse of this theorem is the following one:

:::{prf:theorem}
:label: thm-inverse-k-min

If a graph $G$ with $d+2$ or more vertices does not have a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then any generic {prf:ref}`framework <def-framework>` $p\in C^d(G)$ is not globally rigid.

{{references}} {cite:p}`Gortler2010`
:::

So we can conclude that global rigidity is a generic property.

Since the deterministic algorithm is not very efficient, in the code we use a polynomial-time
randomize algorithm, which will answer "False" all the time if the graph is not generically 
globally rigid in $\mathbb{R}^d$, and it will answer "True" at least half the time if the 
graph is generically globally rigid in $\mathbb{R}^d$.

:::{prf:theorem}
:label: thm-globally-randomize-algorithm

The randomize algorithm for checking global rigidity never returns a false "True" answer, 
and returns a false "False" answer with probability bounded above by $ve/N$, where $v$ is the
number of vertices, $e$ is the number of edges and $N$ is an arbitrarily large integer. 
In this case, we chose $N = 2\cdot ve + 2$ so that the probability of getting a false "False"
is less than 0.5.
In particular, checking for generic global rigidity in $\mathbb{R}^d$ is in $RP$, i.e., 
the class of randomized polynomial time algorithms.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`
{{references}} {cite:p}`Gortler2010`
:::