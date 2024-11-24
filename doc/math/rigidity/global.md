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

Let $G$ be a graph, if $\Omega$ is an {prf:ref}`equilibrium stress matrix <def-stress-matrix>`, 
its kernel is called _stress kernel_; we denote it by $K(\Omega)$ and its dimension by $k(\Omega)$.
We denote by $k_{min}(G,d)$ the minimal value of $k(\Omega)$ as $\Omega$ ranges over all 
{prf:ref}`equilibrium stress matrices <def-stress-matrix>` of all 
{prf:ref}`generic $d$-dimensional frameworks <def-gen-realization>` of $G$.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:lemma}
:label: lem-k-min-stress-matrix

For {prf:ref}`frameworks <def-framework>` of a graph $G$ with at least $d+1$ vertices, 
it holds $k_{min}(G,d) \geq d+1$.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:definition}
:label: def-has-min-stress-kernel

A graph $G$ has a _minimal {prf:ref}`stress kernel <def-stress-kernel>` in $\mathbb{R}^d$_ 
if $k_{min}(G,d) = d+1$.

{{references}} {cite:p}`Gortler2010`
:::

:::{prf:theorem}
:label: thm-k-min-stress-matrix

If a graph $G$ with $d+2$ or more vertices has a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then all {prf:ref}`generic frameworks <def-gen-realization>` $p$ of $G$ are globally rigid.

{{references}} {cite:p}`Gortler2010`
:::

The converse of this theorem is the following one:

:::{prf:theorem}
:label: thm-inverse-k-min-stress-matrix

If a graph $G$ with $d+2$ or more vertices does not have a minimal {prf:ref}`stress kernel <def-stress-kernel>`
in $\mathbb{R}^d$, then any {prf:ref}`generic framework <def-gen-realization>` $p$ of $G$ is not globally rigid.

{{references}} {cite:p}`Gortler2010`
:::
The method {{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid` uses the following randomized algorithm:

Let $d$ be the dimension for which we want to test whether the graph is globally $d$-rigid, 
$v$ be the number of vertices, $e$ be the number of edges, 
$t = v\cdot dim - \binom{dim+1}{2}$ and $N = 10000\cdot v\cdot \binom{v}{2} +2$.
To check if a graph with at least $d + 2$ vertices is generically globally rigid in $\RR^d$, 
proceed as follows:
* If $e < t$, output `False` (as the graph cannot even be generically locally rigid with so few edges), otherwise continue.
* Pick a framework with integer coordinates randomly chosen from 1 to $N$.
* Pick one equilibrium stress vector in a suitably random way.(If $e = t$, there are no stresses, so we consider the zero vector.) 
* Consider the corresponding equilibrium stress matrix and compute its rank. 
* If the rank is $v-dim-1$, return `True`, otherwise return `False` .

:::{prf:theorem}
:label: thm-globally-randomize-algorithm

The randomized algorithm for checking global rigidity never returns a false "True" answer, 
and returns a false "False" answer with probability bounded above by $ve/N$, where $v$ is the
number of vertices, $e$ is the number of edges and $N$ is an arbitrarily large integer. 
In this case, we chose $N = 2\cdot ve + 2$ so that the probability of getting a false "False"
is less than 0.5.
In particular, checking for generic global rigidity in $\mathbb{R}^d$ is in $RP$, i.e., 
the class of randomized polynomial time algorithms.

{{pyrigi_crossref}} {meth}`~.Graph.is_globally_rigid`
{{references}} {cite:p}`Gortler2010`
:::