# Generalized double bananas

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

Suppose $G$ is a flexible $\mathcal{R}_d$-circuit with at most $d+6$ vertices. 
Then either

* $d=3$ and $G\in \{B_{3,2}\}\cup \mathcal{B}_{3,2}^+$ or
* $d\geq 4$ and $G\in \{B_{d,d-1}$, $B_{d,d-2}\}\cup \mathcal{B}_{d,d-1}^+$.

{cite:p}`Grasegger2022`
:::


:::{prf:definition} t-sum
:label: def-t-sum

Given three graphs $G=(V,E)$, $G_1=(V_1,E_1)$, and $G_2=(V_2,E_2)$, we say that
$G$ is a _$t$-sum_ of $G_1,G_2$ along an edge $e$ if $G=(G_1\cup G_2)-e$,
$G_1\cap G_2=K_t$ and $e\in E_1\cap E_2$.
:::


:::{prf:lemma}
:label: lem-2-sum

Suppose that $G=(V,E)$ is the $2$-sum of $G_1=(V_1,E_1)$ and $G_2=(V_2,E_2)$. 
Then $G$ is an $\mathcal{R}_d$-circuit if and only if $G_1$ and $G_2$ are both 
$\mathcal{R}_{d}$-circuits. 

{cite:p}`Grasegger2022`
:::


Using this lemma we can create one family of generalised bananas in which 
every element of the family is a $\mathcal{R}_d$-circuit.

A different generalisation is as follows.

Let $\mathcal{M}=(E,r)$ be a matroid with finite ground set $E$ and rank function $r$.
A _circuit_ of $\mathcal{M}$ is a set $C\subseteq E$ such that $r(C)=|C|-1=r(C-e)$ 
for all $e\in E$. Jackson, Nixon and Smith {cite:p}`Jackson2024` introduced a 
generalisation to _$k$-fold circuits_ i.e. sets $D\subseteq E$ such that 
$r(D)=|D|-2=r(D-e)$ for all $e\in D$, for some fixed integer $k\geq 0$. 

:::{prf:lemma}
:label: lem-k-sum

Let $k\geq 1$ be an integer and let $G$ be the graphical 2-sum of two graphs $G_1$ and
$G_2$ along an edge $e$.
Suppose that $e$ is not a coloop in either $\mathcal{R}_d(G_1)$ or $\mathcal{R}_d(G_2)$.
Then $G$ is a $k$-fold circuit in $\mathcal{R}_d$ if and only if $G_1$ is a $k_1$-fold 
$\mathcal{R}_d$-circuit and $G_2$ is a $k_2$-fold $\mathcal{R}_d$-circuit for some 
$k_1,k_2\geq 1$ with $k_1+k_2=k+1$.

{cite:p}`Jackson2024`
:::

:::{prf:definition} B-d-d-1
:label: def-B-d-d-1

Define $\overline B_{d,d-1}$ to be obtained from $B_{d,d-1}$ by adding back the edge $e$.
It follows that $\textrm{cl} (B_{d,d-1})=\overline B_{d,d-1}$, and so 
$\overline B_{d,d-1}$ is a flexible 2-fold circuit in $\mathcal{R}_d$.
We define the triple banana $B^{(3)}_{d,d-1}$ to be the $2$-sum of $\overline B_{d,d-1}$
and $K_{d+2}$ again along $e$.
By `Lemma <lem-k-sum>`, $B^{(3)}_{d,d-1}$ is a 2-fold circuit in $\mathcal{R}_d$.
Iterating this process, we get that the $(k+1)$-tuple banana $B^{(k+1)}_{d,d-1}$ is a 
$k$-fold circuit in $\mathcal{R}_d$.
:::