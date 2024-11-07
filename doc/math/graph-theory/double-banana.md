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
:::
